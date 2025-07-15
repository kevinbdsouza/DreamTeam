import torch, os, time, math, pickle, numpy as np
from contextlib import nullcontext
from model import GPT, GPTConfig
import time
from torch.cuda.amp import GradScaler, autocast # `autocast` is general for device types, `GradScaler` is also usable with MPS

def train():
    # --- should not change -----------------------------
    t_start = time.time()

    # --- hyper-params not to be changed -----------------------------
    batch_size = 8
    block_size = 64
    n_layer = 4
    n_head = 4
    n_embd = 32
    max_iters = 10001 # Total number of batches to process

    # plain data loader (pre-load into RAM once), not to be changed
    data_dir = "/Users/kevindsouza/Documents/Obsidian_Vault/Companies/SymboliaLabs/research/DreamTeam/data"
    train_data = torch.from_numpy(
        np.memmap(os.path.join(data_dir, "shakespeare/train.bin"), dtype=np.uint16, mode='r')
    ).to(torch.long)
    val_data = torch.from_numpy(
        np.memmap(os.path.join(data_dir, "shakespeare/val.bin"), dtype=np.uint16, mode='r')
    ).to(torch.long)

    # not to be changed
    device = 'mps'
    ctx = nullcontext() # This will be potentially overridden by autocast context

    # -------------------------------------------------------------------
    # anything below this can be changed

    learning_rate = 3e-4 # Initial learning rate
    weight_decay = 0.1 # Parameter for AdamW, originally fixed in the optim call
    betas = (0.9, 0.95) # Parameters for AdamW, originally fixed in the optim call

    # Von Neumann's Numerical and Architectural Enhancements:

    # 1. Gradient Accumulation: Simulates a larger effective batch size by accumulating gradients
    #    over multiple mini-batches before performing an optimizer step. This can lead to more
    #    stable gradient estimates, similar to having more "data points" for each update,
    #    without requiring more memory for a single large batch.
    gradient_accumulation_steps = 4 # Perform optimizer step every X batches
    # Effective batch size becomes batch_size * gradient_accumulation_steps

    # 2. Learning Rate Schedule (Cosine Annealing): A mathematically more sophisticated approach
    #    to learning rate management. It allows for a higher initial learning rate for faster
    #    exploration and then gracefully decreases it to a minimum for fine-tuning convergence,
    #    often leading to better final performance. This aligns with principles of optimization theory.
    lr_decay_iters = max_iters # The total number of iterations over which the LR decays
    lr_min_ratio = 0.1 # Learning rate decays to 10% of its initial value

    # 3. Mixed Precision Training: Utilizes lower precision floating-point numbers (e.g., float16)
    #    for calculations where possible. This can significantly reduce memory bandwidth usage
    #    and speed up computations, especially on hardware accelerators. While MPS devices handle
    #    float32 well, float16 can still offer benefits by reducing memory footprint and potentially
    #    enabling faster transfers. A GradScaler is used to prevent numerical underflow issues
    #    that can arise with float16.
    use_mixed_precision = True
    scaler = None # Initialize scaler to None
    if use_mixed_precision:
        # autocast dynamically switches between FP32 and FP16/BF16 types based on operation.
        # For MPS, float16 is usually the target lower precision type.
        ctx = autocast(device_type=device, dtype=torch.float16)
        scaler = GradScaler() # Used for safe training with float16 to prevent gradient underflow


    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    # Configure optimizer with new learning_rate, weight_decay, betas variables
    optim = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate,
                                       betas=betas, device_type=device)

    # Determine total effective optimizer steps for learning rate schedule
    total_optimizer_steps = math.ceil(max_iters / gradient_accumulation_steps)

    best_vloss = np.inf
    # Keep track of the number of optimizer steps taken
    optimizer_step_counter = 0

    for it in range(max_iters): # 'it' is the batch iteration counter
        # Update learning rate based on cosine annealing schedule
        if lr_decay_iters > 0 and total_optimizer_steps > 0:
            # Current_optimizer_step_progress maps [0, total_optimizer_steps-1] to [0, 1]
            progress = min(1.0, optimizer_step_counter / (total_optimizer_steps))
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
            new_lr = learning_rate * lr_min_ratio + (learning_rate * (1 - lr_min_ratio)) * coeff
            for param_group in optim.param_groups:
                param_group['lr'] = new_lr

        model.train()
        X, Y = get_batch('train')

        # Forward pass and loss calculation under autocast context
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # Scale loss for gradient accumulation

        # Backward pass: compute gradients
        if use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Perform optimizer step only after accumulating gradients for `gradient_accumulation_steps` batches
        if (it + 1) % gradient_accumulation_steps == 0:
            if use_mixed_precision:
                scaler.unscale_(optim) # Unscale gradients before clipping them
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Apply gradient clipping
                scaler.step(optim) # Optimizer step
                scaler.update() # Update the scale for the next iteration
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            optim.zero_grad(set_to_none=True) # Zero gradients efficiently
            optimizer_step_counter += 1 # Increment optimizer step counter

        # Validation check and print happens every 100 *batches*, as specified.
        # This aligns with the original logic for `it % 100 == 0`.
        if (it + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                with ctx: # Use autocast for validation too
                    Xv, Yv = get_batch('val')
                    _, vloss = model(Xv, Yv)
            # Log current LR for insight
            current_lr = optim.param_groups[0]['lr']
            # Report the unscaled training loss for consistency with typical reporting
            print(f'{it+1}: train {loss.item() * gradient_accumulation_steps:.3f}  val {vloss.item():.3f} (LR: {current_lr:.2e})')
            if vloss < best_vloss:
                best_vloss = vloss

    # Ensure any remaining accumulated gradients from a partial final step are applied
    # This handles cases where max_iters is not perfectly divisible by gradient_accumulation_steps
    if (max_iters % gradient_accumulation_steps != 0):
        if use_mixed_precision:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        optim.zero_grad(set_to_none=True)


    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
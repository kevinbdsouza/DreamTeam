import torch, os, time, math, pickle, numpy as np
from contextlib import nullcontext
from model import GPT, GPTConfig
import time


def train():
    # --- should not change -----------------------------
    t_start = time.time()


    # --- hyper-params not to be changed -----------------------------
    batch_size = 8
    block_size = 64
    n_layer = 4
    n_head = 4
    n_embd = 32
    max_iters = 10001

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
    ctx = nullcontext()

    # -------------------------------------------------------------------
    # anything below this can be changed

    # Euler's Optimization Principles Applied:
    # 1. Learning Rate Schedule: A constant 'learning_rate' risks oscillating near the minimum.
    #    By diminishing the step size as we progress, we converge more smoothly and precisely.
    #    A cosine annealing schedule offers a harmonious decay.
    # 2. Gradient Accumulation: With a small 'batch_size', the gradient estimate can be noisy.
    #    Accumulating gradients over several mini-batches provides a more stable and accurate
    #    direction for optimization, much like averaging multiple measurements before a precise adjustment.

    initial_learning_rate = 3e-4 # Our peak learning rate for the early vigorous descent
    final_learning_rate = initial_learning_rate * 0.05 # The learning rate at the very end, for fine-tuning
    warmup_iters = 200 # A brief warmup period to stabilize initial gradients
    grad_accum_steps = 4 # Accumulate gradients over this many batches before an optimization step

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=initial_learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    # To track the actual loss for printing, as the loss for backward pass is scaled
    accumulated_loss_for_logging = 0.0 

    for it in range(max_iters):
        # Adjust the learning rate according to a principled schedule
        if it < warmup_iters:
            # Linear warmup: gradually increase learning rate from zero
            current_lr = initial_learning_rate * (it / warmup_iters)
        else:
            # Cosine decay: smoothly reduce learning rate towards the final value
            decay_progress = (it - warmup_iters) / (max_iters - warmup_iters)
            # Ensure decay_progress does not exceed 1.0, though max_iters handles it generally
            decay_progress = min(1.0, decay_progress) 
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            current_lr = final_learning_rate + (initial_learning_rate - final_learning_rate) * coeff
        
        # Apply the calculated learning rate to the optimizer's parameter groups
        for param_group in optim.param_groups:
            param_group['lr'] = current_lr

        # Set model to training mode
        model.train()
        X, Y = get_batch('train')
        
        # Compute loss and scale it by accumulation steps to correctly average gradients
        with ctx:
            _, loss = model(X, Y)
        loss = loss / grad_accum_steps # Normalize loss for accumulation
        loss.backward()
        
        # Accumulate loss for later logging (unscaled)
        accumulated_loss_for_logging += loss.item() * grad_accum_steps # Rescale back for display

        # Perform optimizer step and zero gradients only when accumulation is complete
        if (it + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Guard against exploding gradients
            optim.step()
            optim.zero_grad(set_to_none=True) # Prepare for the next accumulation cycle
            
            # Reset accumulated loss for the next logging interval
            accumulated_loss_for_logging = 0.0

        # Log training and validation loss every 100 *effective* steps
        # or at the very last iteration to ensure final state is logged
        if ((it + 1) % (100 * grad_accum_steps) == 0) or ((it + 1) == max_iters):
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            
            # Print current iteration, current learning rate, and the observed losses
            # Note: The training loss printed here is the average over the last
            # (100 * grad_accum_steps) iterations if not the final log.
            current_train_loss_for_log = accumulated_loss_for_logging / ( (it + 1) % (100 * grad_accum_steps) if (it + 1) % (100 * grad_accum_steps) != 0 else (100 * grad_accum_steps) ) if accumulated_loss_for_logging != 0 else (loss.item() * grad_accum_steps) # Handle case where it's exactly the checkpoint iteration
            print(f'{it+1}: train {current_train_loss_for_log:.3f} (LR: {current_lr:.2e}) val {vloss.item():.3f}')
            
            # Update best validation loss
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
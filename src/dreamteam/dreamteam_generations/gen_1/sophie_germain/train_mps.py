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

    # Base learning rate for the schedule
    learning_rate = 3e-4

    # Sophie Germain's considerations:
    # A system finds its optimal state not through constant, unyielding force,
    # but through a measured reduction of impetus, akin to damping vibrations
    # to reach equilibrium. We introduce a cosine decay for the learning rate,
    # starting with a brief warm-up phase to establish initial momentum.
    warmup_iters = 100  # Number of iterations for linear warm-up
    lr_decay_iters = max_iters  # Decay learning rate across the entire training duration
    min_lr_ratio = 0.1  # Decay learning rate down to 10% of the base learning rate

    def get_lr(it):
        # 1) Linear warm-up phase
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) If beyond decay iterations, return the minimum learning rate
        if it > lr_decay_iters:
            return learning_rate * min_lr_ratio
        # 3) Cosine decay in between: a smooth progression towards equilibrium
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        # Cosine coefficient ranges from 1.0 down to 0.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return learning_rate * (min_lr_ratio + coeff * (1.0 - min_lr_ratio))

    # Sophie Germain's considerations:
    # To reduce the impact of small, noisy fluctuations (much like microscopic stresses
    # on a material), it is beneficial to average observations. Gradient accumulation
    # simulates a larger batch size, leading to more stable gradient estimates and
    # a smoother, more predictable path to convergence.
    gradient_accumulation_steps = 4  # Accumulate gradients over 4 batches

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # Sophie Germain's considerations:
    # Just as a material benefits from elasticity to withstand stress without fracturing,
    # our model needs a mechanism to prevent it from becoming overly rigid and brittle (overfitting).
    # Dropout introduces a measured amount of 'flexibility' or 'randomness' during training,
    # encouraging more robust feature learning and improving generalization.
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.1)).to(device) # Enabled dropout for robustness
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Determine and set the learning rate for this iteration using the cosine schedule
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
            # Scale the loss by the accumulation factor for proper gradient averaging
            loss = loss / gradient_accumulation_steps

        # Backward pass for gradient accumulation
        loss.backward()

        # Only step the optimizer and zero gradients every `gradient_accumulation_steps`
        # This ensures the accumulated gradients are applied, and the optimizer states are updated.
        if (it + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients for stability
            optim.step()
            optim.zero_grad(set_to_none=True) # Zero gradients for the next accumulation cycle

        # Evaluation and logging every 100 iterations (as per constraint)
        if it % 100 == 0:
            model.eval() # Set model to evaluation mode
            with torch.no_grad(): # Disable gradient calculations for efficiency
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Print the *actual* training loss before scaling by accumulation steps for clarity
            print(f'{it}: train {loss.item() * gradient_accumulation_steps:.3f}  val {vloss.item():.3f} (lr={lr:.1e})')
            if vloss < best_vloss:
                best_vloss = vloss

    # Ensure any remaining accumulated gradients are applied after the loop finishes
    if (max_iters) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
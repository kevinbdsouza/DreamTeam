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

    # Maxwell's touch: Introducing a dynamic learning rate schedule (akin to annealing)
    # and a principled stopping criterion based on observational feedback.

    # The `learning_rate` given is now interpreted as the maximum learning rate.
    learning_rate_max = 3e-4 # Peak learning rate for the schedule
    warmup_iters = int(0.1 * max_iters) # Warm-up phase for initial exploration (10% of total iterations)
    min_lr = learning_rate_max * 0.1 # Minimum learning rate to decay to (10% of max)

    # Early stopping parameters: To find the optimal "equilibrium state"
    # We monitor the validation loss and cease training if no significant improvement is observed.
    eval_interval = 100 # Frequency of validation checks, as per existing constraint
    patience_intervals = 5 # Number of validation intervals to wait for improvement
    patience_iters = patience_intervals * eval_interval # Total iterations to wait
    
    iters_since_last_improvement = 0
    # best_vloss initialized to infinity to ensure any initial validation loss is an improvement.
    best_vloss_overall = np.inf # The overall best validation loss found, for return value
    # To manage patience: we track the best loss since the last time we considered "patience"
    best_vloss_for_patience = np.inf


    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # Cosine learning rate scheduler: Mimics a controlled physical annealing process.
    # Allows for broad exploration initially, then fine-tuning as the system "cools."
    def get_lr(it):
        # 1) Linear warm-up: Gradually increase learning rate from 0 to max_lr
        if it < warmup_iters:
            return learning_rate_max * it / warmup_iters
        # 2) If iteration exceeds max_iters, learning rate should be at its minimum
        if it > max_iters:
            return min_lr
        # 3) Cosine decay: Smoothly reduce learning rate from max_lr to min_lr
        # The decay ratio determines where we are in the cosine curve.
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        # Factor based on cosine wave: 0.5 * (1 + cos(pi * x)) goes from 1 to 0 over x from 0 to 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate_max - min_lr)


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    # Optimizer initialized with the maximum learning rate. Its LR will be updated per iteration.
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate_max,
                                       betas=(0.9, 0.95), device_type='mps')

    # Assigning the initial best_vloss for the return value
    best_vloss = best_vloss_overall

    for it in range(max_iters):
        # Apply the dynamic learning rate to the optimizer for the current iteration.
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        # Standard training step
        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevents "explosions" in the field
        optim.step()
        optim.zero_grad(set_to_none=True)

        # Observation and adaptation: Critically evaluating the system's state.
        # This occurs at regular intervals to gauge performance on unseen data.
        if it % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            current_vloss = vloss.item() # Convert tensor to scalar for comparison and storage
            print(f'{it}: train {loss.item():.3f}  val {current_vloss:.3f} (lr {lr:.2e})')

            # Update overall best validation loss for final return.
            if current_vloss < best_vloss_overall:
                best_vloss_overall = current_vloss

            # Early stopping logic:
            # If the current validation loss is better than the best recorded for patience window, reset counter.
            if current_vloss < best_vloss_for_patience:
                best_vloss_for_patience = current_vloss
                iters_since_last_improvement = 0 # System has found a new, better equilibrium point
            else:
                # No improvement, increment counter.
                iters_since_last_improvement += eval_interval # We add the interval, not just 1

            # If the system has not improved for 'patience_iters', it has stabilized or stagnated.
            # Time to cease further "excitation."
            if iters_since_last_improvement >= patience_iters:
                print(f"Validation loss has not improved for {patience_iters} iterations. Early stopping to prevent overfitting.")
                break # Exit the training loop

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss_overall, elapsed_min
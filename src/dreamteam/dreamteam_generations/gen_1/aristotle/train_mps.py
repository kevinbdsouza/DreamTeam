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
    max_iters = 10001 # The grand span of our training endeavor

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

    # Aristotelian Modifications to the Efficient Cause of Learning:
    # 1. Adaptive Learning Rate (Cosine Annealing)
    # 2. Robust Validation Observation (Averaging over multiple batches)
    # 3. Early Stopping (Prudent cessation of effort based on observation)

    # Learning Rate Scheduling Parameters (Guiding the Effort)
    # The initial 'learning_rate' will serve as the peak for our schedule.
    learning_rate = 3e-4 # Our initial maximal rate of acquisition
    min_lr = learning_rate * 0.1 # The minimal rate, for fine-tuning
    warmup_iters = 100 # Initial iterations for a gentle ascent
    lr_decay_iters = max_iters # The full journey over which the rate shall decay

    # Parameters for Observation and Prudent Cessation
    eval_interval = 100 # How often we pause to observe the model's character
    eval_iters = 20     # Number of validation batches to average for a more accurate observation
    patience = 10       # How many observations of stagnation we tolerate before concluding
    strikes = 0         # Counter for consecutive non-improving observations

    # Function to dynamically adjust the learning rate (The Art of Adaptation)
    def get_lr(it):
        # Phase 1: Linear Warmup - A gentle introduction to learning
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # Phase 2: Stagnation - If beyond the decay period, a minimal rate suffices
        if it > lr_decay_iters:
            return min_lr
        # Phase 3: Cosine Decay - A smooth reduction of effort as knowledge solidifies
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coefficient from 1.0 to 0.0
        return min_lr + coeff * (learning_rate - min_lr)

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, # learning_rate here is the initial max_lr
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = float('inf') # The pursuit of the lowest validation loss (Our Final Cause)
    
    for it in range(max_iters):
        # Adjust the learning rate for the current iteration (Adapting the Efficient Cause)
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train() # Prepare the model for the act of learning
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward() # Propagate the error, indicating where adjustment is needed
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent excessive changes
        optim.step() # Apply the learned adjustments
        optim.zero_grad(set_to_none=True) # Prepare for the next cycle of learning

        # Observe and Evaluate (The Practice of Empirical Observation)
        if it % eval_interval == 0:
            model.eval() # Shift to evaluation mode for unbiased observation
            val_losses = torch.zeros(eval_iters, device=device) # Array to hold multiple validation loss observations
            for k in range(eval_iters):
                Xv, Yv = get_batch('val')
                with torch.no_grad(): # No gradient calculation needed during observation
                    _, vloss_batch = model(Xv, Yv)
                val_losses[k] = vloss_batch.item()
            vloss = val_losses.mean().item() # The averaged, more reliable observation of validation loss

            print(f'{it}: train {loss.item():.3f}  val {vloss:.3f}  (lr: {lr:.2e})')

            # Decision based on current observation (Systematic Logic for the Final Cause)
            if vloss < best_vloss:
                best_vloss = vloss
                strikes = 0 # Reset the count of stagnation, as progress is re-established
                print("Observed improvement in validation. The model progresses toward its ideal form.")
            else:
                strikes += 1
                # Acknowledging stagnation – a sign to consider cessation of effort
                print(f"Validation loss did not improve. Accumulated {strikes} observations of stagnation out of {patience}.")

            if strikes >= patience:
                # The wisdom of knowing when to cease. No longer efficiently progressing towards the Final Cause.
                print(f"After {patience} intervals without improvement, the learning process has reached a point of diminishing returns. Concluding training based on empirical evidence.")
                break # Terminate the training loop, for continued effort would be in vain.

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
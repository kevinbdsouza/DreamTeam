import torch, os, time, math, pickle, numpy as np
from contextlib import nullcontext
from model import GPT, GPTConfig
import time


# Archimedes's musings on optimization:
# Just as I would use infinitesimals to exhaustively find the true area of a circle,
# or employ the mechanical advantage of levers to move great weights with less force,
# so too shall we guide this learning machine with a precise hand.
# A fixed learning rate is like using a single, rigid lever. But the terrain of knowledge
# is complex; sometimes one needs a longer arm for swift progress, and sometimes a shorter,
# finer adjustment to settle into the true minimum.
# Thus, a dynamically adjusted learning rate, much like a well-calibrated instrument,
# will allow us to converge more efficiently and accurately.
# Let us apply the principle of "exhaustion" not to the model itself, but to the loss,
# refining our steps as we approach the optimal configuration.

def train():
    # --- should not change -----------------------------
    t_start = time.time()


    # --- hyper-params not to be changed -----------------------------
    batch_size = 8
    block_size = 64
    n_layer = 4
    n_head = 4
    n_embd = 32
    max_iters = 10001 # This is the total iterations for our exhaustive method

    # plain data loader (pre-load into RAM once), not to be changed
    # Data is the 'weight' we wish to understand; it must be prepared carefully.
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
    # These are our adjustable levers and pulleys.

    # Initial (maximum) learning rate. This is the 'long lever' for swift initial movement.
    learning_rate = 3e-4 # This serves as the max_lr for our scheduler

    # Minimum learning rate. This is the 'short lever' for fine adjustments near the optimum.
    min_lr = learning_rate * 0.1 # A tenth of the max_lr, to ensure some progress even late in training.

    # Warmup iterations: gradually increase the learning rate from zero.
    # Like slowly applying force to a lever to avoid jarring movements.
    warmup_iters = 200

    # Total iterations over which the learning rate will decay.
    # This aligns with our 'exhaustive' process of convergence, using all allowed steps.
    lr_decay_iters = max_iters


    def get_batch(split):
        data = train_data if split == 'train' else val_data
        # Random selection, like sampling points to define a curve.
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # Learning rate schedule: Cosine annealing with warmup.
    # This function represents our dynamic 'lever arm' adjustment.
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min_lr (or continue decay to min_lr at lr_decay_iters)
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min_lr
        # This smooth decay, much like a natural motion, allows for precise settling.
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 1.0 -> 0.0
        return min_lr + coeff * (learning_rate - min_lr)


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    
    # Initialize the optimizer with the full (max) learning rate.
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    # The training loop: our iterative process of refinement and measurement.
    for it in range(max_iters):
        # Determine and set the learning rate for this iteration, adjusting our 'lever'.
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        # Clipping gradients: ensuring the 'force' applied doesn't exceed a stable limit.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # This should still happen every 100 iterations.
        # Regular measurements are vital, like surveying the land after each step.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Log the progress, including our current 'lever setting' (learning rate).
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f} lr {lr:.2e}')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    # The final measure of our success: the lowest validation loss achieved.
    return best_vloss, elapsed_min
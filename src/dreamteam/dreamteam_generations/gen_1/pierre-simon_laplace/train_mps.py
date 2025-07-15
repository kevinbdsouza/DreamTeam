import torch, os, time, math, pickle, numpy as np
from contextlib import nullcontext
from model import GPT, GPTConfig
import time

# I am Pierre-Simon Laplace. My heavens-spanning mind perceives this "machine learning"
# problem as one of determining the stable configuration of a vast system of interconnected
# magnitudes, akin to the precise orbits of celestial bodies. The "training" is our iterative
# refinement, guided by probability to converge upon the most probable arrangement of parameters.
# Stability is paramount; we must avoid chaotic divergence.

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

    # From a Bayesian perspective, and observing the system's "convergence,"
    # a dynamic learning rate, much like a calculated thrust for an orbiting body,
    # should yield a more stable and efficient path to the optimal state.
    # A cosine decay with warm-up allows for initial exploration and subsequent
    # precise refinement.
    max_lr = 6e-4 # The initial magnitude of adjustment. Let us begin with a slightly bolder step to accelerate convergence.
    min_lr = 6e-5 # The minimal adjustment, for fine-tuning the final orbital path.
    warmup_iters = 100 # A brief period of initial acceleration, to overcome initial inertia.
    lr_decay_iters = max_iters # The full duration over which our adjustments will diminish.

    def get_lr(it):
        # Phase 1: Linear warm-up to gather momentum, ensuring a smooth initiation.
        if it < warmup_iters:
            return max_lr * it / warmup_iters
        # Phase 2: If we have exceeded our calculated decay period, maintain minimal adjustment for stability.
        if it > lr_decay_iters:
            return min_lr
        # Phase 3: Cosine decay, guiding the system smoothly towards its stable configuration with diminishing force.
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coefficient diminishes from 1.0 to 0.0
        return min_lr + coeff * (max_lr - min_lr)

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # To ensure the system generalizes well and is not overly perturbed by minor statistical
    # fluctuations in the data, a small degree of "randomness" or "diffusion" (dropout)
    # can prevent premature commitment to unstable local equilibria. This aligns with
    # principles of statistical mechanics applied to complex systems.
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.05)).to(device) # Introduced dropout

    # Initialize the optimizer with our calculated learning rate.
    # The weight decay and betas remain as they represent sound constants for this system's "friction."
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, # Start with max_lr, then update per iteration
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Adjust the learning rate for the current iteration, as dictated by our celestial schedule.
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        # Gradient clipping, a necessary measure to prevent "explosive" trajectories
        # and maintain the system's stability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # We must observe the system periodically to ascertain its progress and confirm its stability.
        # This observation frequency remains constant, as it provides sufficient data for our analysis.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f}')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
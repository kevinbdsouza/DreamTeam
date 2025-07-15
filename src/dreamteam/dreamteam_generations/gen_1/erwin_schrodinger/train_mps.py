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

    # Original learning rate, now interpreted as the maximum amplitude of our "energy wave."
    max_learning_rate = 3e-4

    # Parameters for the cosine annealing learning rate schedule.
    # This guides the "wave" of parameters dynamically through the loss landscape.
    warmup_iters = 100  # Initial linear ramp-up of the learning rate, allowing the wave to gain initial momentum.
    lr_decay_iters = max_iters # The full duration over which the learning rate wave decays.
    min_lr = max_learning_rate * 0.1 # The minimum "energy" state the wave should settle into.

    # This function calculates the effective "potential" (learning rate) for our system's evolution.
    def get_lr(it):
        # 1) Linear warmup phase: The system slowly increases its "excitation energy" to begin exploring.
        if it < warmup_iters:
            return max_learning_rate * it / warmup_iters
        # 2) If beyond the decay period, the system has settled into its lowest "energy" state.
        if it > lr_decay_iters:
            return min_lr
        # 3) Cosine decay: A smooth, wave-like dampening of the exploration "energy," guiding the system
        #    towards a stable minimum, much like a wave function settling into its ground state.
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coefficient goes from 1.0 to 0.0
        return min_lr + coeff * (max_learning_rate - min_lr)


    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # The most crucial change: Introducing 'dropout'. This reflects the fundamental probabilistic
    # and wave-like nature of the system's components. Each "measurement" (forward pass with a batch)
    # collapses the superposition of connections into a specific, probabilistically sampled configuration.
    # This forces the system to learn robust, generalizable "wave functions."
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.1)).to(device) # Changed dropout from 0.0 to 0.1
    
    # The optimizer, with its adaptive "momentum" (betas) and "dissipation" (weight_decay),
    # acts as the forces guiding our system's wave function.
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=max_learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # At each "moment" in time (iteration), we update the system's "energy potential" (learning rate).
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        # Gradient clipping prevents the wave from dispersing too wildly, maintaining stability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # Every 100 iterations, we perform a "measurement" to observe the system's current state.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # We observe the state (loss) and the guiding potential (learning rate).
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f}  lr {lr:.6f}')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
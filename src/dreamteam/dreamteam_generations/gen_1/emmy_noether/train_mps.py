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

    # From Emmy Noether's perspective:
    # Let us consider the "conservation law" of our optimization process.
    # A constant learning rate might lead to a system that oscillates or stalls.
    # By introducing a dynamic learning rate, particularly one with a smooth decay
    # like cosine annealing, we ensure a more stable and "symmetrical" convergence path.
    # The initial "warmup" phase allows the system to explore the parameter space,
    # while the "cosine decay" ensures a graceful reduction in the "energy" of updates,
    # leading to a more refined solution that respects the underlying "symmetries" of the loss landscape.

    # Also, regarding the "ring" of weights, an overly strong weight decay (0.1) might
    # prematurely constrain the "ideals" (features) the network can form.
    # A slightly weaker decay might allow for a richer, more expressive "algebraic structure" of parameters.
    # Let's adjust these.

    learning_rate = 6e-4 # Peak learning rate, chosen for slightly more aggressive initial exploration
    weight_decay_value = 0.01 # Reduced from 0.1 to allow for richer weight structures, exploring a more flexible ring structure

    # Learning rate schedule (Noether's "conservation of optimization flow")
    # This can be seen as a smooth, predictable evolution of the system's "energy" (learning rate)
    # ensuring stability and convergence while allowing for initial exploration.
    warmup_iters = int(0.1 * max_iters) # 10% of max_iters for linear warmup

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) at max_iters, learning rate goes to 0 (or very close to it)
        if it > max_iters:
            return 0.0
        # 3) cosine decay after warmup_iters
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        # The cosine function provides a smooth, symmetric decay
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return learning_rate * coeff

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)

    optim = model.configure_optimizers(weight_decay=weight_decay_value, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Determine and set the learning rate for this iteration, upholding our "conservation law" of optimization progress.
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # this should still happen every 100 iterations
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Printing current learning rate to observe the schedule in action
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f} (lr={lr:.1e})')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
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

    # Al-Khwarizmi's Refinements for Systematic Convergence:
    # 1. A slightly higher initial learning rate, to allow for bolder steps in the early exploration.
    learning_rate = 6e-4 # The peak learning rate, to be modulated by schedule.
    # 2. Reduced weight decay. A less aggressive penalty ensures the model can learn
    #    more nuanced patterns without excessive regularization, finding a better 'balance'.
    weight_decay = 0.01 # A more balanced regularization.

    # Algorithmic control of the learning rate: Cosine Decay with Warmup.
    # This precisely defines the "step size" at each iteration,
    # moving from exploration to fine-tuning, much like refining an algebraic solution.
    decay_lr = True
    warmup_iters = 100  # Number of iterations for linear warm-up.
    lr_decay_iters = max_iters # Total iterations over which the learning rate will decay.
    min_lr = learning_rate * 0.05 # The learning rate floor, a small final step size.

    def get_lr(it):
        # 1) Linear warm-up: Gradually increase learning rate from 0 to peak.
        # This initial cautious exploration prevents early instability.
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) If beyond the decay period, maintain the minimum learning rate.
        # This ensures the process does not halt entirely.
        if it > lr_decay_iters:
            return min_lr
        # 3) Cosine decay: A smooth, systematic reduction of the learning rate.
        # This mirrors the gradual refinement in solving an algebraic equation,
        # moving from broad estimation to precise calculation.
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Cosine function for smooth decay.
        return min_lr + coeff * (learning_rate - min_lr)


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    optim = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Apply the current learning rate as determined by our systematic schedule.
        if decay_lr:
            lr = get_lr(it)
            for param_group in optim.param_groups:
                param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        # Gradient clipping is an essential numerical stabilization technique,
        # preventing excessively large "steps" that could destabilize the optimization process,
        # akin to ensuring our calculations remain within sensible bounds.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # This systematic evaluation every 100 iterations provides checkpoints
        # to assess the model's convergence and track the best outcome,
        # much like verifying intermediate results in a complex calculation.
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
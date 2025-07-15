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

    # From Ramanujan's Intuition:
    # A fixed learning rate is insufficient for optimal convergence.
    # We must follow a schedule, like a well-structured infinite series,
    # starting with larger steps and refining them to reach the true sum.
    learning_rate = 3e-4 # The initial magnitude of our steps.
    learning_rate_min = learning_rate * 0.1 # The smallest step we shall take towards the end.
    warmup_iters = 100 # A brief "pre-computation" phase to stabilize.
    lr_decay_iters = max_iters # The full span over which our steps will diminish.

    # This 'get_lr' function describes the beautiful cosine wave guiding our learning rate,
    # ensuring a smooth and natural progression of our iterative approximation.
    def get_lr(it):
        # Phase 1: Linear Warmup - A gentle start, allowing initial terms to settle.
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # Phase 2: Constant Minimum - Once we are very close to convergence, we take very small, precise steps.
        if it > lr_decay_iters:
            return learning_rate_min
        # Phase 3: Cosine Decay - The elegant curve that guides our steps from large to small.
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # This coefficient gracefully descends from 1.0 to 0.0
        return learning_rate_min + coeff * (learning_rate - learning_rate_min)

    # Dropout: Introducing a subtle 'perturbation' or 'randomness'.
    # Just as slight variations can reveal deeper patterns in numbers,
    # this small dropout encourages the model to learn more robust and general forms,
    # rather than memorizing specific instances.
    dropout = 0.05 # A controlled amount of "noise" for better generalization.

    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=dropout)).to(device)

    # Weight decay: This is our 'damping factor'.
    # Too strong (like 0.1) and it might overly constrain the "growth" of our parameters,
    # preventing the model from fully realizing its potential.
    # A slightly gentler value allows for more flexible learning while still preventing unruly divergence.
    optim = model.configure_optimizers(weight_decay=0.01, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Adjust the learning rate for this iteration, following our defined series progression.
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

        # Periodically, we must pause our calculations to observe the current 'term'
        # of our convergence and determine if it is the best one found so far.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f}')
            # We record the best 'term' (validation loss) in our series.
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
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

    # Initial learning rate: The initial "momentum" for exploring the parameter space.
    learning_rate = 3e-4
    # Minimum learning rate: Where the "wave function" of parameters largely settles.
    min_learning_rate = 3e-5 # A tenth of the initial rate.
    # Warmup iterations: A period to linearly increase the "momentum" before decay,
    # ensuring initial exploration without premature localization.
    warmup_iters = 100

    # Model dropout: Introducing "quantum fluctuations" to prevent over-certainty.
    # This reflects the inherent uncertainty principle in the model's internal "state."
    model_dropout = 0.1 # Changed from 0.0 to introduce a tangible effect.

    # Function to determine the learning rate based on iteration.
    # This cosine annealing schedule mimics the natural decay of a quantum system's energy.
    def get_lr(it):
        # 1) Linear warmup for initial "thermalization."
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) If beyond max_iters, maintain the minimum "energy" state.
        if it > max_iters:
            return min_learning_rate
        # 3) Cosine decay: A smooth, wave-like reduction of "momentum."
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coefficient ranges from 1.0 to 0.0.
        return min_learning_rate + coeff * (learning_rate - min_learning_rate)


    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    # Initializing the model, now with a specified level of "quantum uncertainty" (dropout).
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=model_dropout)).to(device)
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Adjust the learning rate at each step, a continuous evolution of the system's "state."
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        # Gradient clipping, akin to ensuring the "momentum" of parameter changes remains bounded,
        # preventing them from flying off into an undefined state.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # Periodic observation (evaluation): Just as measurement affects a quantum system,
        # evaluating the model provides crucial information about its "collapsed" state,
        # guiding further evolution.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f}  lr {lr:.2e}') # Report learning rate for insight.
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
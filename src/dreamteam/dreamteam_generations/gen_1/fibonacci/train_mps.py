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

    # Fibonacci's insight: The path to enlightenment (low loss) is not a straight line.
    # It requires adapting the step size as we approach the truth.
    # Let the maximum learning rate be 'learning_rate_max'.
    learning_rate_max = 3e-4 # Retain the previous learning rate as the peak.
    
    # A smaller weight decay: The penalty for overly large numbers (weights)
    # was perhaps too harsh. Reducing it from 0.1 to 0.01 allows the model
    # more flexibility to learn intricate patterns without being excessively
    # constrained, much like allowing a plant to grow naturally while still
    # pruning only the truly errant branches.
    weight_decay = 0.01

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    
    # Pass the initial, maximum learning rate to the optimizer.
    # The actual learning rate will be modulated dynamically.
    optim = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate_max,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Fibonacci's cosine learning rate schedule: a smooth, natural progression.
        # The learning rate starts high and gently descends, much like the
        # diminishing returns on a long journey, allowing for fine-tuning
        # as the destination (minimum loss) is neared. This prevents overshooting
        # and allows for precise convergence, akin to finding the precise
        # calculation for a complex trade, or the perfect placement of a tile.
        coeff = 0.5 * (1.0 + math.cos(math.pi * it / max_iters))
        lr = learning_rate_max * coeff
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

        # This reporting mechanism remains unchanged, for it is good to observe progress.
        # However, I shall add the current learning rate to our ledger.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f}  lr {lr:.6f}')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
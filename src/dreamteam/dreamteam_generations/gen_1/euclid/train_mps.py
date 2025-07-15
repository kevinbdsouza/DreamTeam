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

    # Euclid's adjustment: The learning rate, much like the precision of a compass,
    # must be carefully chosen. For this relatively compact construction (model size),
    # a slightly larger yet still measured step can lead to quicker and more accurate convergence.
    learning_rate = 5e-4 # Adjusted from 3e-4 to 5e-4

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # Euclid's adjustment: We introduce a slight perturbation, a "dropout,"
    # akin to adding a degree of resilience to our geometric forms. This prevents
    # them from becoming overly specialized to individual training examples,
    # thus enhancing their generality and robustness (improving validation performance).
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.1)).to(device) # Adjusted dropout from 0.0 to 0.1
    
    # Euclid's adjustment: The weight decay, or the "penalty for complexity,"
    # was perhaps too stringent (0.1). A more moderate penalty (0.01) allows
    # our construction to take on necessary intricacy without becoming overly
    # constrained, finding a better balance between simplicity and accuracy.
    optim = model.configure_optimizers(weight_decay=0.01, learning_rate=learning_rate, # Adjusted weight_decay from 0.1 to 0.01
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
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
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f}')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
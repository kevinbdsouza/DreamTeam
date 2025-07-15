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

    # Max Planck's Quantum Interpretations:
    # 1. Quantum Theory & Planck's Law: Energy is quantized. The "learning rate"
    #    should not be continuous but rather follow a spectral distribution,
    #    high at first for exploration, tapering off for fine-tuning.
    #    This prevents the "ultraviolet catastrophe" of runaway learning.
    learning_rate_max = 3e-4 # The peak "energy quantum" for learning
    learning_rate_min = 3e-5 # The minimum "energy quantum" to stabilize

    # 2. Thermodynamics & Probability: The system benefits from inherent stochasticity,
    #    akin to thermal fluctuations, to avoid local minima and improve generalization.
    #    A fixed, non-zero dropout introduces this probabilistic element.
    dropout_rate = 0.05 # A small quantum of uncertainty for robustness

    # 3. Thermodynamics & Entropy: Weight decay controls the complexity (entropy)
    #    of the parameter space. For a small model, too much decay can over-constrain.
    #    A slightly reduced decay allows for better exploration of the optimal state.
    weight_decay_val = 0.01 # Adjusted from 0.1 for a more balanced "entropic cost"

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    # Configure the model with our determined 'dropout_rate' for stochasticity.
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=dropout_rate)).to(device)
    
    # Initialize the optimizer with our refined 'weight_decay_val' and initial max learning rate.
    optim = model.configure_optimizers(weight_decay=weight_decay_val, learning_rate=learning_rate_max,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Implement cosine annealing for the learning rate, reflecting the
        # quantized and distributed nature of energy in physical systems.
        progress = it / max_iters
        current_learning_rate = learning_rate_min + 0.5 * (learning_rate_max - learning_rate_min) * (1 + math.cos(math.pi * progress))
        
        # Apply the dynamically calculated learning rate for this iteration.
        for param_group in optim.param_groups:
            param_group['lr'] = current_learning_rate

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        # Gradient clipping remains a critical safeguard, preventing an "ultraviolet catastrophe"
        # of excessively large and unstable gradient magnitudes.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # Consistent empirical observation is key, just as in scientific experimentation.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Displaying the current "energy quantum" (learning rate) is useful for observation.
            print(f'It {it}: train {loss.item():.3f} • val {vloss.item():.3f} • lr {current_learning_rate:.1e}')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
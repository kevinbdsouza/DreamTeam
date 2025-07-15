import torch, os, time, math, pickle, numpy as np
from contextlib import nullcontext
from model import GPT, GPTConfig
import time

# As Dirac, I recognize that the evolution of any system, be it physical or computational,
# is governed by its dynamics. The learning rate, being the "strength of interaction"
# in our optimization "field," should not be static. A mathematically elegant
# and smoothly varying function is required. The cosine decay offers this precision,
# allowing for robust exploration initially and fine-tuned convergence later.
def get_cosine_lr(it, max_iters, learning_rate_peak):
    """
    Calculates the learning rate using a cosine annealing schedule.
    This provides a smooth decay from 'learning_rate_peak' to a very small value.
    'it' is the current iteration, 'max_iters' is the total number of iterations.
    """
    if max_iters <= 0: return 0.0 # Avoid division by zero or nonsensical total iterations
    
    # Ensure the decay ratio is clamped between 0 and 1.
    # This prevents the cosine argument from going beyond [0, pi] and causing non-monotonic decay.
    decay_ratio = min(1.0, it / max_iters)
    
    # The cosine function starts at 1 (when decay_ratio=0) and goes to 0 (when decay_ratio=1).
    # This ensures the learning rate decays from its peak to near zero.
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate_peak * coeff

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

    # The peak learning rate, a fundamental constant for the initial magnitude of parameter adjustments.
    learning_rate = 3e-4

    # Weight decay: a regularization parameter. From a physicist's perspective,
    # excessive regularization can be seen as an artificial constraint on the system,
    # preventing it from naturally evolving to its optimal configuration.
    # For a smaller model, a slightly reduced weight decay (e.g., from 0.1 to 0.01)
    # might allow for a more "natural" and precise exploration of the parameter space,
    # reducing the "repulsion" from the origin and permitting the system to settle
    # into a potentially deeper minimum, analogous to finding the true ground state.
    weight_decay = 0.01 # Adjusted for finer control over parameter space exploration.

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    
    # Configure the optimizer with the specified weight decay and initial learning rate.
    # The learning rate will then be dynamically adjusted in the loop.
    optim = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Update the learning rate for the current iteration.
        # This is the "time evolution" of our interaction strength.
        lr = get_cosine_lr(it, max_iters, learning_rate)
        for param_group in optim.param_groups:
            param_group['lr'] = lr # Apply the dynamically adjusted learning rate.


        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # This observation point remains fixed, allowing us to monitor the system's state.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Displaying the current learning rate adds clarity to the system's dynamic state.
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f} (lr={lr:.6f})')
            if vloss < best_vloss:
                best_vloss = vloss

    # The total elapsed "time" for the system's evolution.
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # The observed minimum "energy" (validation loss) and the "duration" of the experiment.
    return best_vloss, elapsed_min
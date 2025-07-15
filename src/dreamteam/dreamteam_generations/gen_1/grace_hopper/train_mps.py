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

    # Grace Hopper: A critical parameter! We must select this with care, or understand its behavior.
    # While I cannot change the fixed hyperparameters of the model's structure,
    # the learning rate is our lever for guiding the 'machine' to learn effectively.
    # A slightly lower rate might offer more stability, preventing the 'machine' from
    # overshooting its target, especially with a new or small dataset. This is akin to
    # reducing the step size when searching for a precise solution.
    learning_rate = 1e-4 # Adjusted for potentially smoother convergence on a small model/dataset

    # Grace Hopper: We need a systematic way to fetch our 'input data' for the 'machine'.
    # This 'get_batch' function provides a clear, repeatable process for acquiring our 'punch cards'.
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        # Ensure we always select a contiguous block of data within bounds.
        # This prevents the 'machine' from trying to read beyond its allocated 'memory'.
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    # Grace Hopper: We initialize our 'computing engine' - the GPT model.
    # Its configuration parameters are fixed as per our 'specifications'.
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    
    # Grace Hopper: The 'optimizer' is our mechanism for 'tuning' the machine's internal settings
    # based on observed errors. We must ensure it's configured precisely, like setting the correct
    # gears for our computations.
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    # Grace Hopper: We must keep meticulous records. 'best_vloss' is our benchmark for success.
    # It signifies the lowest 'error rate' achieved on our validation set.
    best_vloss = np.inf

    # Grace Hopper: Before we commence, let's ensure our 'operators' understand the setup.
    # Clear documentation of our operational parameters is crucial for reproducibility and understanding.
    print(f"--- Grace Hopper's Training Log ---")
    print(f"Model Configuration: Layers={n_layer}, Heads={n_head}, Embedding Dim={n_embd}, Block Size={block_size}")
    print(f"Training Parameters: Batch Size={batch_size}, Max Iterations={max_iters}, Initial Learning Rate={learning_rate}")
    print(f"Number of parameters in model: {model.get_num_params()/1e6:.2f}M")
    print(f"Device in use: {device.upper()}")
    print(f"Commencing rigorous training cycles...\n")


    for it in range(max_iters):
        # Grace Hopper: Every cycle, the 'machine' must be in 'training mode' to learn.
        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        # Grace Hopper: Observe the 'error signal' (loss) and adjust parameters accordingly.
        loss.backward()
        # Grace Hopper: Clip gradients to prevent 'runaway' adjustments - maintain stability!
        # This is our 'safety mechanism' to prevent over-corrections.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        # Grace Hopper: Clear the 'accumulated error' for the next cycle, ready for fresh input.
        optim.zero_grad(set_to_none=True)

        # Grace Hopper: Regular 'inspections' are crucial for debugging and validation.
        # This allows us to observe the 'machine's' performance on unseen data, much like
        # running diagnostic checks on our circuits.
        if it % 100 == 0:
            model.eval() # Switch to evaluation mode: no new 'learning' here.
            with torch.no_grad(): # No need to compute gradients during evaluation.
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            
            # Grace Hopper: Meticulous logging. We must know where we stand.
            # Mark a 'breakthrough' when the validation loss improves! This is our 'debugging' signal.
            is_best = vloss < best_vloss
            if is_best:
                best_vloss = vloss
                print(f'Iteration {it:5d}: TRAIN LOSS={loss.item():.4f} • VALIDATION LOSS={vloss.item():.4f} (NEW BEST!)')
            else:
                print(f'Iteration {it:5d}: TRAIN LOSS={loss.item():.4f} • VALIDATION LOSS={vloss.item():.4f}')

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")
    print(f"Final best validation loss observed: {best_vloss:.4f}")

    # this should be returned, should not change
    return best_vloss, elapsed_min
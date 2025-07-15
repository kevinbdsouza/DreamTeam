import torch, os, time, math, pickle, numpy as np
from contextlib import nullcontext
from model import GPT, GPTConfig
import time
import torch.optim as optim # Import optim module for scheduler

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

    learning_rate = 3e-4 # Initial learning rate for the scheduler
    gradient_accumulation_steps = 4 # Accumulate gradients over 4 batches to simulate a larger batch size

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    
    # Initialize optimizer with the base learning rate
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    # Learning rate scheduler: Cosine Annealing
    # T_max is the total number of iterations over which the learning rate will decay.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

    best_vloss = np.inf
    for it in range(max_iters):
        model.train()
        X, Y = get_batch('train')
        
        with ctx:
            _, loss = model(X, Y)
            # Scale the loss by gradient_accumulation_steps.
            # This ensures that the effective gradient magnitude for a 'larger batch' is consistent.
            loss = loss / gradient_accumulation_steps
        
        # Backward pass to accumulate gradients
        loss.backward()

        # Only perform optimizer step and zero gradients when accumulation is complete
        # This is where we apply the "averaged" gradient from multiple small batches.
        if (it + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Update learning rate after the optimizer step
            scheduler.step() # Advance the learning rate scheduler

        # This should still happen every 100 iterations.
        # We print the unscaled loss (multiplied by accumulation steps) for better interpretability.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Print the scaled training loss for better reporting (undoing the division for accumulation)
            print(f'Iter {it}/{max_iters-1}: train_loss={loss.item() * gradient_accumulation_steps:.3f}  val_loss={vloss.item():.3f}  lr={optimizer.param_groups[0]["lr"]:.6f}')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
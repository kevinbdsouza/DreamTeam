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
    max_iters = 10001 # This is the *maximum* allowed iterations. Our axiomatic approach will define a termination earlier.

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

    # Initial learning rate. We shall introduce a principled decay.
    learning_rate = 3e-4 # This will be our initial peak learning rate.

    # --- Hilbertian Additions: Axioms for Training Dynamics ---
    # Axiom 1: Principled Learning Rate Schedule (Cosine Annealing)
    # This guides our steps through the parameter space in a mathematically smooth manner,
    # ensuring both exploration and fine-tuning.
    # We define the total number of iterations over which to decay the learning rate.
    lr_decay_iters = max_iters 
    min_lr = 1e-5 # The floor for our learning rate, preventing it from vanishing completely.

    # Axiom 2: Early Cessation (Patience for Validation Loss)
    # We establish a criterion for when the model has sufficiently converged or stagnated.
    # If the validation loss does not improve for `patience` successive validation checks,
    # we declare that further training yields no new "knowledge" according to our system's axioms.
    # A validation check happens every 100 iterations. So, patience of 50 implies 5000 iterations without improvement.
    patience = 50 
    bad_iters = 0 # Counter for consecutive validation checks without improvement.
    # --- End of Hilbertian Additions ---


    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # Function to calculate learning rate based on cosine annealing.
    # This provides a smooth decay from peak_lr to min_lr over lr_decay_iters,
    # adhering to the principle of continuous and well-defined transformations in our mathematical system.
    def get_lr(it):
        # We start decay from iteration 0.
        decay_ratio = (it) / (lr_decay_iters) 
        # Ensure the ratio is clamped between 0 and 1.
        decay_ratio = min(1.0, max(0.0, decay_ratio)) 
        # Cosine coefficient, ranging from 1.0 down to 0.0.
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return min_lr + coeff * (learning_rate - min_lr)


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, # Initial LR passed, will be updated.
                                       betas=(0.9, 0.95), device_type='mps')

    # Initialize best_vloss as a tensor of infinite value on the correct device.
    # This ensures type consistency for comparisons and the final return value, aligning with original behavior.
    best_vloss = torch.tensor(float('inf'), device=device)

    for it in range(max_iters):
        # Update learning rate based on our principled cosine schedule.
        # This is a continuous adjustment of our step size.
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

        # This should still happen every 100 iterations, as per the constraint.
        # This is our periodic evaluation point for the model's axiomatic convergence.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv) # vloss is a scalar tensor
            
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f} (LR: {lr:.2e})')

            # --- Early Stopping Logic (Axiom of Early Cessation) ---
            # We determine if the current validation loss represents an improvement.
            if vloss < best_vloss:
                # If an improvement is found, we update our best observed loss.
                # We clone().detach() to ensure we store a snapshot without retaining computation graph history.
                best_vloss = vloss.clone().detach() 
                bad_iters = 0 # Reset the patience counter, as progress has been made.
            else:
                # If no improvement, we increment the counter of 'unfruitful' validation checks.
                bad_iters += 1 

            # If our patience is exhausted, we declare the training process complete.
            # This is our formally defined termination condition, establishing a finite basis.
            if bad_iters >= patience:
                print(f"Validation loss has not improved for {patience*100} iterations ({patience} checks). Early stopping based on axiomatic patience.")
                break # Exit the training loop


    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
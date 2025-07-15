import torch, os, time, math, pickle, numpy as np
from contextlib import nullcontext
from model import GPT, GPTConfig # Ensure GPTConfig is properly imported for configuration
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
    max_iters = 10001 # The ultimate horizon of iterations, though we may stop sooner

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
    # My Newtonian principles for optimization

    # The maximum "impetus" (learning rate) for vigorous initial exploration
    learning_rate_max = 3e-4
    # The minimum "impetus" for fine-tuning near the optimal state
    learning_rate_min = 3e-5 # One-tenth of the maximum rate
    # A short "warm-up" phase to steadily increase the initial impetus
    warmup_iters = 100
    # The "patience" of our observation: how many evaluation cycles without improvement before stopping
    patience = 5 # Corresponds to 500 iterations (5 * 100 evaluation steps)
    # The rate of "random obscuration" (dropout) to promote robustness and generalization
    dropout_rate = 0.1 # A modest perturbation to prevent over-specialization

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # A function to dynamically adjust the learning rate, following a cosine decay schedule.
    # This ensures a smooth reduction of "impetus" as the system approaches equilibrium.
    def get_lr(it):
        # 1) Linear warmup phase to smoothly increase the initial impetus
        if it < warmup_iters:
            return learning_rate_max * it / warmup_iters
        # 2) If we exceed the total decay iterations, maintain the minimum impetus
        if it > max_iters:
            return learning_rate_min
        # 3) In between, apply a cosine decay, allowing the system to settle gracefully
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        # The coefficient smoothly transitions from 1.0 to 0.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return learning_rate_min + coeff * (learning_rate_max - learning_rate_min)

    # Construct the model, now incorporating the chosen dropout rate
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=dropout_rate)).to(device)
    
    # Configure the optimizer, initially set with the maximum learning rate
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate_max,
                                       betas=(0.9, 0.95), device_type='mps')

    # Initialize variables for tracking the best validation loss and early stopping
    best_vloss = float('inf') # Our initial, infinitely high "energy state"
    patience_counter = 0 # Counter for observed stagnation

    # Begin the iterative process of refinement
    for it in range(max_iters):
        # Adjust the "impetus" (learning rate) for this specific iteration
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train() # Set the model to training mode
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y) # Compute the loss, our measure of "disorder"
        loss.backward() # Propagate the "forces" (gradients) backwards through the system
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent "explosive forces"
        optim.step() # Apply the "forces" to adjust the system's parameters
        optim.zero_grad(set_to_none=True) # Reset the forces for the next iteration

        # Periodically assess the model's ability to generalize, akin to testing a physical law
        # This observation occurs every 100 iterations
        if it % 100 == 0:
            model.eval() # Set the model to evaluation mode (e.g., disable dropout for consistent measurement)
            with torch.no_grad(): # No need to compute gradients during observation
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv) # Observe the validation loss
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f}')

            # Apply the principle of "minimal action": stop if no further significant improvement
            if vloss < best_vloss:
                best_vloss = vloss # A new, better "energy state" has been achieved
                patience_counter = 0 # Reset the counter, as progress is observed
            else:
                patience_counter += 1 # The system appears to be settling, or stagnating
                if patience_counter >= patience:
                    print(f"Validation loss has not improved for {patience} consecutive checks ({patience*100} iterations). Stopping training early as per the principle of efficiency and minimal action.")
                    break # Halt the process, for further effort yields little gain.

    # Final reporting of the system's performance and the time consumed in its optimization
    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # This is the final and best observed validation loss, and the total elapsed time.
    # It shall be returned as dictated.
    return best_vloss, elapsed_min
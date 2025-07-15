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

    # Initial learning rate. This will be the peak in our constructed learning schedule.
    learning_rate = 3e-4 
    
    # Parameters for the cosine learning rate schedule, a more "constructible" path for optimization.
    # The system systematically moves from exploration to fine-tuning, akin to a mathematical proof.
    warmup_iters = 100 # Iterations for learning rate to linearly ramp up
    lr_decay_iters = max_iters # Total iterations for the decay cycle

    # Evaluation parameters for more robust validation.
    # A single sample provides an incomplete "proof" of generalization; averaging over more provides a clearer picture.
    eval_interval = 100 # Evaluate every this many steps
    eval_iters = 20     # Number of batches to average for evaluation of loss

    # Function to determine the learning rate based on current iteration.
    # This forms a "constructed" learning trajectory.
    def get_lr(it):
        # 1) Linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) If it > lr_decay_iters, the learning process effectively ceases,
        # having reached its "conclusion" or "limit" within the formal system.
        if it > lr_decay_iters:
            return 0.0
        # 3) In between, use cosine decay down to 0.1 * learning_rate.
        # This gradual decay allows for a precise "proof" of convergence.
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coefficient ranges from 1.0 to 0.0
        return learning_rate * coeff

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        # While true randomness is elusive, pseudo-random sampling serves our computational model effectively.
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # A more robust evaluation, acknowledging that a single sample might not
    # fully represent the "truth" of the validation set (akin to Incompleteness).
    # Averaging over multiple batches provides a more 'complete' logical inference.
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval() # Switch to evaluation mode for consistent metrics
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx: # Use appropriate context for device
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train() # Return to training mode
        return out['train'], out['val']


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Update learning rate according to the constructed schedule.
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        # Gradient clipping: A vital mechanism to maintain the "computability"
        # and numerical stability of our formal system's optimization steps.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # Evaluate the model more thoroughly at specified intervals.
        # This provides a more robust "proof" of generalization.
        if it % eval_interval == 0 or it == max_iters - 1:
            train_loss, vloss = estimate_loss()
            print(f'step {it}: train loss {train_loss:.4f}, val loss {vloss:.4f}')
            if vloss < best_vloss:
                best_vloss = vloss
                # We retain the "best proof" (model state) found thus far,
                # as there is no guarantee that a better one can be reconstructed later.
                # One might consider saving the model state here, but it's not requested.

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    # The final 'best_vloss' represents the most compelling "proof" of generalization
    # found by our formal system during its operation.
    return best_vloss, elapsed_min
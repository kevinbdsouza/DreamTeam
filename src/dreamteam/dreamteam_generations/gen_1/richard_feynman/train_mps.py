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

    # The learning rate: This is like the "energy" we give our particle to explore the landscape.
    # A fixed energy is not efficient. We need to adjust it dynamically.
    learning_rate = 3e-4 # This will now be our maximum learning rate.
    min_lr = learning_rate * 0.1 # Let's not go to zero; leave a little 'thermal energy' for exploration.
    warmup_iters = 100 # A gentle linear warm-up, like slowly accelerating a particle.
    lr_decay_iters = max_iters # The full decay will span the entire training.

    # Gradient accumulation: Instead of acting on a single, noisy "measurement" (batch),
    # we'll gather information from multiple "measurements" before making a move.
    # This averages out the "quantum fluctuations" in the gradient, giving us a clearer "path direction."
    gradient_accumulation_steps = 4 # Effectively quadrupling our batch size without using more memory at once.

    # This function determines the "energy" (learning rate) for our particle at each step.
    # It starts low, warms up, then slowly decays, allowing broad exploration then fine-tuning.
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps, preventing early instability
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, we've settled, return minimum learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay, guiding our particle towards the minimum
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # This smooth curve is like a gentle push
        return min_lr + coeff * (learning_rate - min_lr)

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, # learning_rate here is the initial max_lr
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Determine and set the learning rate for this iteration, dynamically!
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        
        # Scale the loss for gradient accumulation. This averages the gradient over multiple steps,
        # like taking a more precise measurement before adjusting our trajectory.
        loss = loss / gradient_accumulation_steps 
        
        loss.backward() # Compute the gradients (the "force" guiding our particle)

        # Only step the optimizer and zero gradients after accumulating enough.
        # This is when we actually move the "particle" in parameter space.
        if (it + 1) % gradient_accumulation_steps == 0 or it == max_iters - 1:
            # Clip the gradients to prevent "runaway" paths or instabilities.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step() # Take a step in the direction of the accumulated gradient
            optim.zero_grad(set_to_none=True) # Clear the forces for the next measurement cycle

        # this should still happen every 100 iterations
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # When logging, we'll show the "true" batch loss, scaled back up
            print(f'{it}: train {loss.item() * gradient_accumulation_steps:.3f}  val {vloss.item():.3f}')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
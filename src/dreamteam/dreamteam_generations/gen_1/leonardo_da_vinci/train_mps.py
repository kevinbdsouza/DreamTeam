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
    max_iters = 10001 # This must remain the upper bound for iterations

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

    # The maximum learning rate, which will now be decayed
    learning_rate = 3e-4

    # Parameters for the learning rate schedule (cosine decay with warmup)
    warmup_iters = 100 # Initial iterations for linear learning rate ramp-up
    lr_decay_iters = max_iters # Total iterations over which learning rate decays
    min_lr = learning_rate * 0.1 # Minimum learning rate to decay to (1/10th of max)

    # Function to calculate the dynamic learning rate based on iteration
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min_lr
        #    (This handles cases where the loop runs beyond the decay schedule, though here lr_decay_iters = max_iters)
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        # Ensure ratio is within bounds for cosine calculation
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 1.0 -> 0.0
        return min_lr + coeff * (learning_rate - min_lr)


    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # Introducing a small dropout for regularization, as observed in nature's designs
    # for robustness against overfitting.
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.1)).to(device) # Dropout changed to 0.1
    
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Determine and set the learning rate for this iteration based on cosine decay
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

        # This should still happen every 100 iterations for observation
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Print current status, much like logging observations in a notebook
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f} (lr={lr:.1e})')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
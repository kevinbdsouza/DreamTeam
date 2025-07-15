
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
    train_data = torch.from_numpy(
        np.memmap('data/shakespeare/train.bin', dtype=np.uint16, mode='r')
    ).to(torch.long)
    val_data = torch.from_numpy(
        np.memmap('data/shakespeare/val.bin', dtype=np.uint16, mode='r')
    ).to(torch.long)

    # not to be changed
    device = 'mps'
    ctx = nullcontext()

    # -------------------------------------------------------------------
    # anything below this can be changed

    # Initial learning rate for the "exploration" phase
    learning_rate = 3e-4

    # Define the learning rate schedule for the "refinement" phase (cosine decay)
    lr_decay_iters = max_iters # The total iterations over which learning rate decays
    min_lr = 1e-5 # The minimum learning rate to avoid stagnation
    warmup_iters = 100 # Steps for linear warmup at the beginning

    # Gradient accumulation to effectively simulate a larger batch size,
    # reducing the "uncertainty" in gradient estimates before updating parameters.
    gradient_accumulation_steps = 4

    def get_lr(it):
        # Linear warmup phase
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # If beyond decay period, return minimum learning rate
        if it > lr_decay_iters:
            return min_lr
        # Cosine decay phase
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 1.0 to 0.0
        return min_lr + coeff * (learning_rate - min_lr)

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # Introducing a measure of "uncertainty" in the model's internal states
    # via dropout, promoting a more robust and generalizable "quantum" model.
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.1)).to(device)
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Adjust the learning rate based on our defined schedule
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        # Accumulate gradients over several micro-steps before updating
        # to get a more stable gradient estimate.
        for micro_step in range(gradient_accumulation_steps):
            X, Y = get_batch('train')
            with ctx:
                _, loss = model(X, Y)
                # Scale the loss to account for gradient accumulation,
                # ensuring the average loss per sample is consistent.
                loss = loss / gradient_accumulation_steps
            loss.backward()

        # Apply gradient clipping after all micro-steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # this should still happen every 100 iterations
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                # To reduce the "uncertainty" in our validation measurement,
                # average the loss over multiple validation batches.
                val_losses = []
                num_val_batches_for_avg = 10 # Number of batches to average for validation loss
                for _ in range(num_val_batches_for_avg):
                    Xv, Yv = get_batch('val')
                    _, vloss_single = model(Xv, Yv)
                    val_losses.append(vloss_single.item())
                vloss = sum(val_losses) / len(val_losses)
            # Print the effective training loss (scaled back) and validation loss
            print(f'{it}: train {loss.item() * gradient_accumulation_steps:.3f}  val {vloss:.3f}')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min


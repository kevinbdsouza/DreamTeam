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
    max_iters = 10001 # Total cycles for the computation engine to operate

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

    # Babbage's Enhanced Control Parameters
    initial_learning_rate = 3e-4 # The maximum speed for parameter adjustments
    warmup_iters = 100 # Cycles for the engine to 'warm up' to full adjustment speed
    lr_decay_iters = max_iters # The total span over which the adjustment speed will decay
    min_lr_factor = 0.1 # The fraction of initial_learning_rate to decay to (e.g., 0.1 means 10%)
    patience_threshold = 10 # Number of periodic validation checks without improvement before halting

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=initial_learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    # Babbage's Rate of Adjustment Governor (Learning Rate Scheduler)
    def get_lr(it_current):
        lr_max = initial_learning_rate
        lr_min = initial_learning_rate * min_lr_factor

        # 1) Linear warmup phase
        if it_current < warmup_iters:
            return lr_max * it_current / warmup_iters
        # 2) If beyond decay iterations, maintain minimum learning rate
        if it_current > lr_decay_iters:
            return lr_min
        # 3) Cosine decay phase between warmup and end of decay
        decay_ratio = (it_current - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coeff goes from 1.0 to 0.0
        return lr_min + (lr_max - lr_min) * coeff

    best_vloss = float('inf') # The finest observed reading on the validation table
    patience_counter = 0 # Counter for consecutive periodic inspections without improvement

    for it in range(max_iters):
        # Adjust the learning rate for this cycle, using the governor mechanism
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

        # This should still happen every 100 iterations (Periodic inspection of the machine's progress)
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            current_vloss = vloss.item() # Read the current value from the validation table
            print(f'{it}: train {loss.item():.3f}  val {current_vloss:.3f}')

            # Babbage's Efficiency Halt Mechanism: Check for improvement
            if current_vloss < best_vloss:
                best_vloss = current_vloss # Record the new finest reading
                patience_counter = 0 # Reset the counter as progress was noted
            else:
                patience_counter += 1 # Increment if no improvement this period

            if patience_counter >= patience_threshold:
                print(f"No further significant improvement in validation loss for {patience_threshold} periodic checks. Halting computation early to conserve resources.")
                break # Cease operations, for efficiency and foresight

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # This should be returned, and should not change
    # Ensure best_vloss is a scalar value for the final report
    if isinstance(best_vloss, torch.Tensor):
        best_vloss = best_vloss.item()
    return best_vloss, elapsed_min
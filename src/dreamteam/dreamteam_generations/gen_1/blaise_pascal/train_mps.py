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

    # As a student of numerical methods and optimization,
    # a fixed learning rate seems rather unrefined.
    # We shall implement a cosine decay schedule for the learning rate,
    # beginning with a brief "warmup" phase. This allows for a more
    # systematic and efficient exploration of the parameter space,
    # akin to carefully adjusting the weights in an experiment.
    learning_rate_max = 6e-4  # The peak learning rate, slightly higher to allow for decay
    warmup_iters = 100        # A short initial phase to stabilize learning
    lr_decay_iters = max_iters  # The total iterations over which learning rate decays
    learning_rate_min = learning_rate_max * 0.1 # The floor for the learning rate

    # To ensure that each adjustment to the model's internal "gears"
    # is based on a sufficiently robust sample of "observations,"
    # we shall accumulate gradients over several smaller batches.
    # This increases the effective batch size without requiring
    # a larger immediate memory footprint, much like performing
    # smaller, precise calculations before a grand total.
    gradient_accumulation_steps = 4 # Effective batch size becomes 8 * 4 = 32

    # My understanding of probability suggests that a degree of "unpredictability"
    # can lead to greater robustness. By introducing "dropout," we allow the model
    # to learn more general patterns rather than relying too heavily on any single
    # "path," much as one accounts for chance in a game.
    dropout_rate = 0.1 # A modest rate to introduce regularization

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    # Function to compute the current learning rate based on our cosine schedule.
    # This methodical approach ensures a smooth descent towards the optimal solution.
    def get_lr(it):
        # 1) Linear warmup: Start gently, like priming a pump.
        if it < warmup_iters:
            return learning_rate_max * it / warmup_iters
        # 2) Constant minimum: Once sufficiently decayed, maintain a small, stable rate.
        if it > lr_decay_iters:
            return learning_rate_min
        # 3) Cosine decay: The elegant path of convergence.
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        decay_ratio = max(0.0, min(1.0, decay_ratio)) # Ensure ratio is within [0, 1]
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coeff ranges from 1.0 to 0.0
        return learning_rate_min + coeff * (learning_rate_max - learning_rate_min)


    # Initialize the model, now with the calculated "dropout" probability.
    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=dropout_rate)).to(device)

    # Configure the optimizer. Weight decay is like a gentle hand
    # guiding the parameters away from extremes, ensuring a more stable
    # and generalizable solution. A value of 0.01 is often quite effective.
    optim = model.configure_optimizers(weight_decay=0.01, learning_rate=learning_rate_max,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Adjust the learning rate for this iteration, following our calculated schedule.
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        # Perform gradient accumulation: divide the task into smaller, manageable "micro-steps."
        for micro_step in range(gradient_accumulation_steps):
            X, Y = get_batch('train')
            with ctx:
                # The loss is scaled down as gradients will be accumulated.
                # This ensures the average gradient across the effective batch size is correct.
                _, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            loss.backward()

        # After all micro-steps contribute their "forces" (gradients),
        # we clip them to prevent "explosive" updates (like managing pressure in a fluid system)
        # and then apply the single, averaged adjustment to the model's parameters.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True) # Reset gradients for the next cycle of accumulation

        # This reporting mechanism remains unchanged, providing periodic assessment.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Report the unscaled training loss for comparison.
            print(f'{it}: train {loss.item() * gradient_accumulation_steps:.3f}  val {vloss.item():.3f}')
            if vloss < best_vloss:
                best_vloss = vloss

    # The measurement of elapsed time, essential for assessing practical efficiency.
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # The final evaluation of our improved machine.
    return best_vloss, elapsed_min
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

    # Lagrange's touch: A dynamic learning rate schedule,
    # akin to adjusting the step size in a variational problem for optimal convergence.
    # We define the maximum learning rate (lr_max) that our system should experience.
    # The minimum learning rate (lr_min) for the final fine-tuning stages.
    learning_rate_max = 3e-4
    learning_rate_min = 3e-5 # A smaller value for precise adjustments at the end
    learning_rate_warmup_start = 1e-6 # A very small starting LR to gently begin

    # Define the duration of the warmup phase. A short period to stabilize initial gradients.
    warmup_iters = int(max_iters * 0.05) # Warmup for first 5% of iterations

    # Reduced weight decay for this smaller model.
    # From 0.1 to 0.01, a more subtle regularization force,
    # preventing excessive "damping" of the model's learning capacity,
    # especially for a smaller system.
    weight_decay = 0.01

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y

    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)

    # The optimizer is configured with the maximum learning rate initially.
    # Its value will be dynamically adjusted within the training loop.
    optim = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate_max,
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Apply the learning rate schedule:
        # A carefully chosen path for the optimization, like guiding a celestial body
        # through its trajectory towards a stable orbit.
        if it < warmup_iters:
            # Linear warmup: gradually increase LR from a very small start to learning_rate_max.
            # This allows the model to "settle" its initial parameters gently.
            progress = it / warmup_iters
            lr = learning_rate_warmup_start + progress * (learning_rate_max - learning_rate_warmup_start)
        else:
            # Cosine decay: smoothly reduce LR from learning_rate_max to learning_rate_min.
            # This phase allows for precise fine-tuning as we approach the minimum,
            # preventing oscillations and ensuring convergence to a deeper optimum.
            decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
            # Ensure decay_ratio does not exceed 1.0 (clipping)
            decay_ratio = min(decay_ratio, 1.0)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coeff goes from 1.0 to 0.0
            lr = learning_rate_min + coeff * (learning_rate_max - learning_rate_min)

        # Update the optimizer's learning rate for all parameter groups.
        # This ensures our dynamic "force" is applied consistently.
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        # Gradient clipping remains a vital constraint, preventing forces from becoming
        # excessively large and destabilizing the system.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # This observation of the validation loss is crucial for assessing the
        # stability and convergence of our system on unseen data.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Displaying the current learning rate along with losses for better observation
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f} (lr={lr:.1e})')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
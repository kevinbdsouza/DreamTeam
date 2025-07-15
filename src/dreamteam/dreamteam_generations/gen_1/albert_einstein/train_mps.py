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

    # The maximum learning rate, this will be scaled down by the schedule.
    # This is the "peak gravitational strength" we wish to apply.
    learning_rate = 3e-4 

    # A number of iterations for the learning rate to linearly increase.
    # This "warmup" period allows the parameters to gently explore the loss landscape,
    # preventing premature "collapse" into suboptimal local minima, akin to
    # a rocket gradually building speed before escaping Earth's gravity.
    warmup_iters = 200 # A modest fraction of total iterations

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps.
        # The "gravitational pull" gradually strengthens from zero to its maximum.
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > max_iters, learning rate becomes negligible.
        # The system has settled; the "gravity" approaches zero.
        if it > max_iters:
            return 0.0 
        # 3) in between, use cosine decay down to 0.
        # This is the "cosmic deceleration": as we approach the "center of gravity"
        # (the loss minimum), the pull reduces smoothly, allowing for fine-tuning
        # and preventing oscillations, much like a particle gently settling into a potential well.
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cosine goes from 1 to 0
        return learning_rate * coeff

    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    # The optimizer is initialized with the peak learning rate, which is then dynamically adjusted.
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, 
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    for it in range(max_iters):
        # Dynamic adjustment of the learning rate: akin to precisely tuning the
        # gravitational constant as our "universe" of parameters evolves.
        # This ensures efficient navigation of the loss landscape.
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        # Gradient clipping prevents "runaway" gradients, ensuring stability.
        # This is like constraining the velocity of particles to prevent them
        # from escaping the gravitational field of the loss landscape.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # This should still happen every 100 iterations. Observation is key.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Printing current learning rate to observe the schedule in action,
            # providing insight into the dynamic "force" guiding the system.
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f} (lr {lr:.2e})')
            if vloss < best_vloss:
                best_vloss = vloss

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # This should be returned, should not change
    return best_vloss, elapsed_min
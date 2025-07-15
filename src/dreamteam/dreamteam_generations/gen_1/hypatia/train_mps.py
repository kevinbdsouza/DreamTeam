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

    # Hypatia's wisdom: Just as celestial motions follow a harmonious path,
    # so too should our model's progression. A fixed learning rate is akin
    # to a static observer; instead, we implement a dynamic schedule.
    # We increase the base learning rate slightly for initial exploration,
    # then carefully decay it for precise convergence.
    learning_rate = 6e-4 # Base learning rate, allowing for more vigorous initial steps.
    
    # Weight decay, analogous to a gentle guiding force, ensures the model
    # remains robust without being overly constrained. A slightly reduced value
    # allows the model to capture more intricate patterns from the data,
    # preventing it from being too rigid.
    weight_decay = 0.05 

    # Parameters for the celestial-inspired learning rate schedule
    warmup_iters = int(max_iters * 0.1)  # A period of warming up, like an initial survey of the night sky.
    lr_decay_iters = max_iters           # The decay continues throughout the training journey.
    min_lr = learning_rate * 0.1         # The learning rate gracefully diminishes to a minimum, for fine-tuning.

    def get_lr(it):
        # 1) Linear warmup for the initial phases, for rapid exploration.
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) If the journey extends beyond the decay period, maintain the minimal pace.
        if it > lr_decay_iters:
            return min_lr
        # 3) In between, the cosine function guides the learning rate, mirroring
        # the elegant and smooth arcs of celestial bodies, leading to harmonious convergence.
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coefficient ranges from 1.0 to 0.0
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
    optim = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate,
                                       betas=(0.9, 0.95), device_type='mps')

    # Initialize best_vloss to a large value, anticipating a path towards lower error.
    best_vloss = float('inf') 
    for it in range(max_iters):
        # Dynamically adjust the learning rate with each iteration,
        # ensuring the model progresses with optimal speed and precision,
        # just as one would adjust an astrolabe for the most accurate reading.
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train()
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)
        loss.backward()
        # Gradient clipping, like setting the boundaries for planetary orbits,
        # prevents extreme deviations and ensures the stability of our parameters.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        # This periodic observation is crucial for tracking progress,
        # akin to validating our astronomical tables against new observations.
        if it % 100 == 0:
            model.eval()
            with torch.no_grad():
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            print(f'{it}: train {loss.item():.3f}  val {vloss.item():.3f} (lr={lr:.1e})')
            # We meticulously record the lowest validation loss,
            # representing our most precise approximation of the truth.
            if vloss.item() < best_vloss: # Convert tensor to float for comparison and storage
                best_vloss = vloss.item()

    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
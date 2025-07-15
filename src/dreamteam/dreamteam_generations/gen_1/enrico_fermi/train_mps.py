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
    max_iters = 10001 # This is effectively the number of optimizer steps

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

    # Initial learning rate. This will be the peak LR in our schedule.
    learning_rate = 3e-4 # Baseline learning rate from the original setup

    # Fermi's Enhancements: Guiding the "chain reaction" for better stability and efficiency

    # 1. Gradient Accumulation (Increasing Effective Critical Mass)
    # We will accumulate gradients over this many micro-batches before performing an optimizer step.
    # This effectively makes our batch size `batch_size * gradient_accum_steps`.
    # A larger effective batch size can lead to more stable gradient estimates.
    gradient_accum_steps = 4 

    # 2. Learning Rate Schedule (Controlled Neutron Flux)
    # Cosine decay down to a small fraction of the initial LR.
    # Warmup to stabilize the initial "chain reaction".
    learning_rate_decay_iters = max_iters # The learning rate will decay throughout the entire training duration
    min_lr = learning_rate * 0.1          # The learning rate will decay down to 10% of its initial value
    warmup_iters = 100                    # Linear warmup for the first 100 optimizer steps

    # Function to calculate the current learning rate based on the iteration number
    def get_lr(it):
        # 1) Linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) If it > learning_rate_decay_iters, return min_lr (plateau)
        if it > learning_rate_decay_iters:
            return min_lr
        # 3) In between, use cosine decay down to min_lr
        decay_ratio = (it - warmup_iters) / (learning_rate_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        # Cosine coefficient ranges from 1.0 (start of decay) to 0.0 (end of decay)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return min_lr + coeff * (learning_rate - min_lr)

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        # Ensure that ix is on the correct device for torch.randint
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
        return x, y


    model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, block_size=block_size,
                          vocab_size=50304, bias=False, dropout=0.0)).to(device)
    optim = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, # Initial LR is passed here
                                       betas=(0.9, 0.95), device_type='mps')

    best_vloss = np.inf
    
    # Training loop: 'it' now represents the optimizer step count
    for it in range(max_iters):
        # Determine and set the current learning rate based on our schedule
        lr = get_lr(it)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        model.train() # Ensure the model is in training mode (e.g., for dropout)

        # Gradient accumulation: Perform multiple forward/backward passes before an optimizer update
        # This increases the effective batch size without increasing memory usage proportionally.
        accumulated_loss_sum = 0.0 # To keep track of the loss over accumulation steps
        optim.zero_grad(set_to_none=True) # Clear gradients at the beginning of each accumulation cycle

        for micro_step in range(gradient_accum_steps):
            X, Y = get_batch('train')
            with ctx:
                _, loss = model(X, Y)
                # Scale the loss by the number of accumulation steps to effectively average gradients
                # This ensures the gradient magnitudes are consistent with a larger single batch.
                loss = loss / gradient_accum_steps 
            loss.backward() # Accumulate gradients

            # Sum the scaled loss for reporting. This will be the average loss per micro-batch.
            accumulated_loss_sum += loss.item() 

        # Clip gradients after all accumulation steps are complete
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Perform the optimizer step: update model weights based on accumulated gradients
        optim.step()

        # Validation and logging: Still happens every 100 optimizer steps as per constraint
        if it % 100 == 0:
            model.eval() # Set model to evaluation mode (e.g., disable dropout)
            with torch.no_grad(): # Disable gradient calculations for efficiency
                Xv, Yv = get_batch('val')
                _, vloss = model(Xv, Yv)
            # Print the averaged training loss over the accumulation steps and current learning rate
            print(f'{it}: train {accumulated_loss_sum:.3f} (lr {lr:.1e}) val {vloss.item():.3f}')
            if vloss < best_vloss:
                best_vloss = vloss
    
    # not to be changed
    elapsed_min = (time.time() - t_start) / 60
    print(f"\nTraining complete • total wall-clock time: {elapsed_min:.1f} min")

    # this should be returned, should not change
    return best_vloss, elapsed_min
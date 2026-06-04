import os
import time
import math
import pickle
from contextlib import nullcontext
import wandb

import numpy as np
import torch
from sklearn.model_selection import KFold

from model import Delphi, DelphiConfig
from utils import get_p2i, get_batch


out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
seed = None 

# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'delphi'
wandb_run_name = 'test_cv' 


# data
dataset = 'ukb_data'
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 24

# model
n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.2  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
vocab_size = 256

# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 10000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 10000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float32'  # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster

# delphi training
token_dropout = 0.0
t_min = 0.0  # 365.25/12.
mask_ties = True
ignore_tokens = [0]
data_fraction = 1.0
no_event_token_rate = 5
k_folds = 10

# pretrained embeddings
adapter_n_layers = 2
adapter_n_hidden = 256
embeddings_path = "data/emb_diseases_m2.npy"  # path to .npy file, or None to train from scratch

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
with open('configurator.py') as f:
    exec(f.read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
if seed is not None:
    torch.manual_seed(seed) 
torch.set_float32_matmul_precision('high')

device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'float64': torch.float64,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_dtype(ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r').reshape(-1, 3)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r').reshape(-1, 3)

all_data = np.concatenate([train_data, val_data], axis=0)
all_p2i = get_p2i(all_data)
n_train_patients = len(get_p2i(train_data)) # for when kfolds= 1 

# downsample the data to requested fraction
if data_fraction < 1.0:
    all_p2i = all_p2i[:int(data_fraction * len(all_p2i))]
patient_indices = np.arange(len(all_p2i))

if k_folds == 1:
    # equivalent to original train/val split
    splits = [(patient_indices[:n_train_patients], patient_indices[n_train_patients:])]
else:
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(patient_indices))

# load pretrained embeddings if a valid path was provided
pretrained_embeddings = None
if embeddings_path is not None:
    pretrained_embeddings = np.load(embeddings_path)




for fold, (train_fold_idx, val_fold_idx) in enumerate(splits):
    print(f"\n========== FOLD {fold+1}/{k_folds} ==========")
    out_dir_fold = os.path.join(out_dir, f'fold_{fold+1}')
    os.makedirs(out_dir_fold, exist_ok=True)
    np.save(os.path.join(out_dir_fold, 'train_ids.npy'), train_fold_idx)
    np.save(os.path.join(out_dir_fold, 'val_ids.npy'), val_fold_idx)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    
    print(f"found vocab_size = {vocab_size}")

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                          bias=bias, vocab_size=vocab_size, dropout=dropout, token_dropout=token_dropout, t_min=t_min,
                          mask_ties=mask_ties, ignore_tokens=ignore_tokens,
                          pretrained_embeddings=pretrained_embeddings,
                          adapter_n_layers=adapter_n_layers,
                          adapter_n_hidden=adapter_n_hidden, pretrained_embd_dim=pretrained_embeddings.shape[1] if pretrained_embeddings is not None else None)  

    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        gptconf = DelphiConfig(**model_args)
        model = Delphi(gptconf, pretrained_embeddings=pretrained_embeddings)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = DelphiConfig(**model_args)
        model = Delphi(gptconf, pretrained_embeddings=pretrained_embeddings)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16')) 

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # helps estimate an arbitrarily accurate loss over either split using many batches


    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters, 2)
            ix_pool = train_fold_idx if split == 'train' else val_fold_idx
            for k in range(eval_iters):
                ix = ix_pool[torch.randint(len(ix_pool), (batch_size,))]
                X, A, Y, B = get_batch(ix, all_data, all_p2i, block_size=block_size,
                                       device=device, select='left',
                                       no_event_token_rate=no_event_token_rate, 
                                       cut_batch=True)
                with ctx:
                    logits, loss, _ = model(X, A, Y, B, validation_loss_mode=True)
                losses[k] = torch.stack([loss['loss_ce'], loss['loss_dt']])
            out[split] = losses.mean(0)
        model.train()
        return out


    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)


    # logging
    if wandb_log:
        wandb.init(project=wandb_project, name=f'{wandb_run_name}_{fold}', group=wandb_run_name, config=config)

    # training loop
    ix = train_fold_idx[torch.randint(len(train_fold_idx), (batch_size,))]
    X, A, Y, B = get_batch(ix, all_data, all_p2i, block_size=block_size, device=device,
                           padding='random', lifestyle_augmentations=True, select='left',
                           no_event_token_rate=no_event_token_rate)
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process

    val_loss = None
    while True:
        metrics = {"iter": iter_num}
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and iter_num > 0:
            losses = estimate_loss()
            val_loss = losses['val'].sum().item()
            print(f"step {iter_num}: train loss {losses['train'].sum().item():.4f}, val loss {val_loss:.4f}")

            if best_val_loss > val_loss:
                best_val_loss = val_loss

            metrics.update({
                "train/agg_loss": losses['train'].sum().item(),
                "val/loss": val_loss,
                "val/loss_ce": losses['val'][0].item(),
                "val/loss_dt": losses['val'][1].item(),
            })

            if always_save_checkpoint or best_val_loss == val_loss:
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir_fold}")
                    torch.save(checkpoint, os.path.join(out_dir_fold, 'ckpt.pt'))

            if iter_num % 10_000 == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir_fold}")
                torch.save(checkpoint, os.path.join(out_dir_fold, f'ckpt_{iter_num}.pt'))

        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss, att = model(X, A, Y, B)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            ix = train_fold_idx[torch.randint(len(train_fold_idx), (batch_size,))]
            X, A, Y, B = get_batch(ix, all_data, all_p2i, block_size=block_size, device=device,
                                   padding='random', lifestyle_augmentations=True, select='left',
                                   no_event_token_rate=no_event_token_rate, cut_batch=True)

            # backward pass, with gradient scaling if training in fp16
            loss_ce_item = loss['loss_ce'].item()
            loss_dt_item = loss['loss_dt'].item()
            loss = loss['loss_ce'] + loss['loss_dt']
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

            metrics.update({
                "train/loss": lossf,
                "train/loss_ce": loss_ce_item,
                "train/loss_dt": loss_dt_item,
                "lr": lr,
            })

        if wandb_log and (iter_num % log_interval == 0 or "val/loss" in metrics):
            wandb.log(metrics)

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break
    if wandb_log:
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.finish()

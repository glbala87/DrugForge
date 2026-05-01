"""DrugForge training script.

Supports DTA (regression), DTI (classification), MoA (classification) modes
with uncertainty-weighted multitask loss, auxiliary descriptor/funcgroup tasks,
and KL beta-annealing.

Usage:
    python train.py [dataset_idx] [gpu_idx]
    python train.py 0          # davis (DTA mode)
    python train.py 1          # kiba  (DTA mode)
    python train.py 2          # bindingdb (DTA mode)
"""

import logging
import os
import pickle
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import ModelConfig, TrainConfig, DatasetThresholds
from dataset import MolProteinDataset
from metrics import (concordance_index_fast, mse, rmse, modified_r2,
                     aupr_at_thresholds, classification_metrics)
from model import DrugForge
from task_balancer import MultiTaskBalancer

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_msg(msg, log_dir, tag):
    with open(os.path.join(log_dir, f'log_{tag}.txt'), 'a') as f:
        f.write(f'{msg}\n')
    logger.debug(msg)


def train_epoch(model, balancer, loader, optimizer, primary_loss_fn,
                epoch, device, tcfg, mcfg):
    model.train()
    balancer.train()

    with tqdm(loader, desc=f'Epoch {epoch + 1}') as pbar:
        for data in pbar:
            optimizer.zero_grad()
            data = data.to(device)

            primary_pred, lm_loss, kl_loss, cl_loss, aux_losses = model(data)

            # Primary loss depends on task mode
            if mcfg.task_mode in ('dti', 'moa'):
                primary_loss = primary_loss_fn(primary_pred, data.y.view(-1, 1).float())
            else:
                primary_loss = primary_loss_fn(primary_pred, data.y.view(-1, 1).float())

            # Gather all losses
            losses = {
                'primary': primary_loss,
                'lm': lm_loss,
                'contrastive': cl_loss,
                'kl': kl_loss,
            }

            # Add auxiliary losses
            for aux_name, aux_loss in aux_losses.items():
                losses[aux_name] = aux_loss

            total_loss, info = balancer(losses, epoch=epoch)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(balancer.parameters(), tcfg.grad_clip)
            optimizer.step()

            pbar.set_postfix(
                Pri=f"{info.get('primary_raw', 0):.3f}",
                LM=f"{info.get('lm_raw', 0):.3f}",
                KL=f"{info.get('kl_raw', 0):.3f}",
                Aux=f"{sum(v for k,v in info.items() if k.endswith('_raw') and k not in ('primary_raw','lm_raw','kl_raw','contrastive_raw')):.3f}",
            )

    return info


@torch.no_grad()
def evaluate(model, loader, dataset_name, device, task_mode='dta'):
    model.eval()
    all_true, all_pred = [], []
    thresholds = DatasetThresholds().get(dataset_name)

    for data in tqdm(loader, desc='Evaluating', leave=False):
        data = data.to(device)
        pred = model.predict(data)
        all_true.append(data.y.view(-1, 1).cpu())
        all_pred.append(pred.cpu())

    y_true = torch.cat(all_true).numpy().flatten()
    y_pred = torch.cat(all_pred).numpy().flatten()

    if task_mode in ('dti', 'moa'):
        return classification_metrics(y_true, y_pred), y_true, y_pred
    else:
        return {
            'mse': mse(y_true, y_pred),
            'rmse': rmse(y_true, y_pred),
            'ci': concordance_index_fast(y_true, y_pred),
            'rm2': modified_r2(y_true, y_pred),
            'aupr': aupr_at_thresholds(y_true, y_pred, thresholds),
        }, y_true, y_pred


def save_checkpoint(model, balancer, optimizer, epoch, best_metric, mcfg, path):
    """Save a full training checkpoint for resume capability."""
    torch.save({
        'model': model.state_dict(),
        'balancer': balancer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric,
        'task_mode': mcfg.task_mode,
    }, path)
    logger.info(f'Checkpoint saved: epoch {epoch + 1} -> {path}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DrugForge Training')
    parser.add_argument('--dataset', type=str, default='bindingdb',
                        choices=['davis', 'kiba', 'bindingdb'])
    parser.add_argument('--task-mode', type=str, default=None,
                        choices=['dta', 'dti', 'moa', 'multi'],
                        help='Override task mode from config')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto-detect)')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--eval-every', type=int, default=None)
    parser.add_argument('--checkpoint-every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    dataset = args.dataset

    if args.device:
        device_str = args.device
    elif torch.cuda.is_available():
        device_str = 'cuda:0'
    else:
        device_str = 'cpu'
    device = torch.device(device_str)

    tcfg = TrainConfig()
    mcfg = ModelConfig()

    # CLI overrides
    if args.task_mode:
        mcfg.task_mode = args.task_mode
    if args.epochs:
        tcfg.num_epochs = args.epochs
    if args.batch_size:
        tcfg.batch_size = args.batch_size
    if args.lr:
        tcfg.learning_rate = args.lr
    if args.eval_every:
        tcfg.eval_every = args.eval_every

    set_seed(tcfg.seed)

    log_dir = os.path.join(args.output_dir, 'logs')
    model_dir = os.path.join(args.output_dir, 'saved_models')
    aff_dir = os.path.join(args.output_dir, 'Affinities')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(aff_dir, exist_ok=True)

    log_tag = f'{dataset}_{int(time.time())}'

    logger.info(f'DrugForge | Dataset: {dataset} | Mode: {mcfg.task_mode} | Device: {device}')
    logger.info(f'Batch: {tcfg.batch_size} | LR: {tcfg.learning_rate} | Epochs: {tcfg.num_epochs}')
    logger.info(f'Morgan FP: {mcfg.use_morgan_fp} | ESM-2: {mcfg.use_esm2}')
    logger.info(f'Aux tasks: descriptors={mcfg.use_descriptor_head}, funcgroups={mcfg.use_funcgroup_head}')

    # Load tokenizer (prefer JSON, fall back to pickle)
    from tokenizer import MolTokenizer
    tok_json = os.path.join(args.data_dir, f'{dataset}_tokenizer.json')
    tok_pkl = os.path.join(args.data_dir, f'{dataset}_tokenizer.pkl')
    if os.path.isfile(tok_json):
        tokenizer = MolTokenizer.load_json(tok_json)
    elif os.path.isfile(tok_pkl):
        logger.warning(f'Loading tokenizer from pickle (insecure): {tok_pkl}')
        with open(tok_pkl, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        logger.error(f'No tokenizer found for {dataset}')
        return

    train_cache = os.path.join(args.data_dir, 'processed', f'{dataset}_train.pt')
    test_cache = os.path.join(args.data_dir, 'processed', f'{dataset}_test.pt')
    if not (os.path.isfile(train_cache) and os.path.isfile(test_cache)):
        logger.error('Preprocessed data not found. Run create_data.py first!')
        return

    train_data = MolProteinDataset(cache_path=train_cache)
    test_data = MolProteinDataset(cache_path=test_cache)
    train_loader = DataLoader(train_data, batch_size=tcfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=tcfg.batch_size, shuffle=False)

    # Initialize model
    model = DrugForge(tokenizer, mcfg).to(device)

    # Task names for balancer
    task_names = ['primary', 'lm', 'contrastive']
    if mcfg.use_descriptor_head:
        task_names.append('descriptor')
    if mcfg.use_funcgroup_head:
        task_names.append('funcgroup')

    balancer = MultiTaskBalancer(
        task_names=task_names,
        kl_max_weight=tcfg.kl_max_weight,
        kl_warmup_epochs=tcfg.kl_warmup_epochs,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Parameters: {total_params:,}')

    optimizer = optim.AdamW(
        list(model.parameters()) + list(balancer.parameters()),
        lr=tcfg.learning_rate,
    )

    # Primary loss function
    if mcfg.task_mode in ('dti', 'moa'):
        primary_loss_fn = nn.BCEWithLogitsLoss()
    else:
        primary_loss_fn = nn.MSELoss()

    best_metric = float('inf') if mcfg.task_mode == 'dta' else 0.0
    start_epoch = 0
    model_path = os.path.join(model_dir, f'drugforge_{dataset}.pth')
    ckpt_path = os.path.join(model_dir, f'drugforge_{dataset}_checkpoint.pth')

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f'Resuming from checkpoint: {args.resume}')
            ckpt = torch.load(args.resume, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model'])
            balancer.load_state_dict(ckpt['balancer'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            best_metric = ckpt.get('best_metric', best_metric)
            logger.info(f'Resumed at epoch {start_epoch}, best_metric={best_metric}')
        else:
            logger.error(f'Checkpoint not found: {args.resume}')
            return

    # Graceful shutdown on SIGINT/SIGTERM
    _shutdown_requested = [False]

    def _signal_handler(signum, frame):
        if _shutdown_requested[0]:
            logger.warning('Forced exit.')
            sys.exit(1)
        _shutdown_requested[0] = True
        logger.info('Shutdown requested. Saving checkpoint after current epoch...')

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    G, P = None, None

    for epoch in range(start_epoch, tcfg.num_epochs):
        info = train_epoch(model, balancer, train_loader, optimizer,
                           primary_loss_fn, epoch, device, tcfg, mcfg)
        log_msg(str(info), log_dir, log_tag)

        if (epoch + 1) % tcfg.eval_every == 0:
            results, G, P = evaluate(model, test_loader, dataset, device, mcfg.task_mode)

            # Save best model
            if mcfg.task_mode in ('dti', 'moa'):
                current = results.get('AUROC', 0)
                is_best = current > best_metric
                if is_best:
                    best_metric = current
            else:
                current = results.get('mse', float('inf'))
                is_best = current < best_metric
                if is_best:
                    best_metric = current

            if is_best:
                torch.save({
                    'model': model.state_dict(),
                    'balancer': balancer.state_dict(),
                    'epoch': epoch,
                    'best_metric': best_metric,
                    'task_mode': mcfg.task_mode,
                }, model_path)
                logger.info('Best model saved!')

            for k, v in results.items():
                if isinstance(v, list):
                    logger.info(f"  {k}: {[f'{a:.4f}' for a in v]}")
                else:
                    logger.info(f"  {k}: {v:.4f}")

            sigmas = [f'{n}={v.item():.3f}' for n, v in balancer.log_vars.items()]
            logger.info(f'  sigma: {", ".join(sigmas)}')

        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(model, balancer, optimizer, epoch, best_metric, mcfg, ckpt_path)

        # Graceful shutdown
        if _shutdown_requested[0]:
            save_checkpoint(model, balancer, optimizer, epoch, best_metric, mcfg, ckpt_path)
            logger.info(f'Stopped at epoch {epoch + 1}. Resume with --resume {ckpt_path}')
            return

    if P is not None and G is not None:
        np.savetxt(os.path.join(aff_dir, f'estimated_{dataset}.txt'), P)
        np.savetxt(os.path.join(aff_dir, f'true_{dataset}.txt'), G)
    logger.info('Training complete.')


if __name__ == '__main__':
    main()

"""Offline training: DIAMOND denoiser only, no Atari environment or RL agent."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from mitocheck_data import TrackIndex, CellWindows
from models.diffusion import Denoiser, DenoiserConfig, InnerModelConfig, SigmaDistributionConfig, DiffusionSampler, DiffusionSamplerConfig


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def worker_seed(_):
    s = torch.initial_seed() % (2**32)
    np.random.seed(s); random.seed(s)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def make_index(cfg):
    return TrackIndex(cfg.dataset.root, cfg.dataset.expected_train, cfg.dataset.expected_test)


def dataset(index, split, cfg):
    return CellWindows(index, split, cfg.dataset.history, cfg.dataset.val_fraction, cfg.seed, cfg.dataset.interval, cfg.dataset.offset)


def make_model(cfg, gene_dim):
    m = cfg.model
    inner = InnerModelConfig(1, cfg.dataset.history, m.cond_channels, list(m.depths), list(m.channels), list(m.attn_depths), cell_condition_dim=gene_dim+3)
    model = Denoiser(DenoiserConfig(inner,m.sigma_data,m.sigma_offset_noise))
    model.setup_training(SigmaDistributionConfig(**OmegaConf.to_container(cfg.sigma)))
    return model


def sampler(model, cfg):
    return DiffusionSampler(model,DiffusionSamplerConfig(**OmegaConf.to_container(cfg.sampler)))


def audit(index, cfg):
    result = dict(index.audit); result['subsets'] = {}
    for split in ('train','val','test'):
        ds = dataset(index,split,cfg)
        result['subsets'][split] = {'tracks':len(ds.track_ids),'windows':len(ds),'track_ids':ds.track_ids}
        for i in np.linspace(0, len(ds)-1, min(cfg.training.audit_samples,len(ds)),dtype=int): ds[int(i)]
    result['history'] = cfg.dataset.history
    result['time_policy'] = {'offset_h':cfg.dataset.offset,'interval_h':cfg.dataset.interval,'age':'source frame minus track start'}
    return result


def validate_contract(saved, index, cfg):
    if saved['contract'] != index.contract:
        raise ValueError('Dataset identity/embedding/fixed membership differs from checkpoint')
    old = OmegaConf.create(saved['config'])
    if OmegaConf.to_container(old.model) != OmegaConf.to_container(cfg.model) or OmegaConf.to_container(old.sigma) != OmegaConf.to_container(cfg.sigma):
        raise ValueError('Model/EDM configuration differs from checkpoint')
    for key in ('history','val_fraction','interval','offset'):
        if old.dataset[key] != cfg.dataset[key]: raise ValueError(f'Dataset policy differs: {key}')
    if old.seed != cfg.seed: raise ValueError('Partition seed differs from checkpoint')


def save_checkpoint(path, model, optimizer, cfg, index, epoch, step, best, membership):
    state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(), 'config':OmegaConf.to_container(OmegaConf.masked_copy(cfg, [k for k in cfg if k not in ('hydra','defaults')]),resolve=True),
             'contract':index.contract,'membership':membership,'epoch':epoch,'step':step,'best_val':best,
             'rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
             'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp=Path(path).with_suffix('.tmp'); torch.save(state,tmp);tmp.replace(path)


def run(cfg):
    if cfg.dataset.interval <= 0: raise ValueError('Positive frame interval required')
    torch.set_num_threads(cfg.cpu_threads); seed_all(cfg.seed)
    out=Path(cfg.output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not cfg.resume: raise ValueError('Output already has a run; choose a new output_dir or resume')
    index=make_index(cfg); report=audit(index,cfg); write_json(out/'data_audit.json',report)
    print(json.dumps({s:{k:v for k,v in d.items() if k!='track_ids'} for s,d in report['subsets'].items()},indent=2))
    if cfg.mode=='audit': return report
    if cfg.mode!='train': raise ValueError('mode must be train or audit')
    dev=torch.device(cfg.device)
    if dev.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; use device=cpu for smoke tests')
    model=make_model(cfg,index.genes.shape[1]).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.training.lr,weight_decay=cfg.training.weight_decay)
    train=dataset(index,'train',cfg); val=dataset(index,'val',cfg)
    if not len(train) or not len(val): raise ValueError('Need nonempty train and train-only validation windows')
    start=0;step=0;best=float('inf')
    if cfg.resume:
        state=torch.load(cfg.resume,map_location='cpu',weights_only=False);validate_contract(state,index,cfg)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        start=state['epoch']+1;step=state['step'];best=state['best_val']
        torch.set_rng_state(state['rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if dev.type=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    membership={s:report['subsets'][s]['track_ids'] for s in ('train','val','test')}
    for epoch in range(start,cfg.training.epochs):
        loader=DataLoader(train,batch_size=cfg.training.batch_size,shuffle=True,num_workers=cfg.training.workers,
                          worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(cfg.seed+epoch),pin_memory=dev.type=='cuda')
        model.train(); total=0.;count=0
        for i,b in enumerate(loader):
            if cfg.training.max_batches is not None and i>=cfg.training.max_batches:break
            opt.zero_grad(set_to_none=True)
            loss=model.cell_loss(b['history'].to(dev),b['target'].to(dev),b['condition'].to(dev))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.training.max_grad_norm);opt.step()
            total+=float(loss.detach())*len(b['target']);count+=len(b['target']);step+=1
        if not count:raise ValueError('No training batches executed')
        model.eval();vt=0.;vc=0
        # Repeat validation noise every epoch for comparable checkpoint selection.
        with torch.random.fork_rng(devices=[dev.index or 0] if dev.type=='cuda' else []):
            torch.manual_seed(cfg.seed+100000)
            with torch.no_grad():
                for i,b in enumerate(DataLoader(val,batch_size=cfg.training.batch_size,num_workers=cfg.training.workers,worker_init_fn=worker_seed)):
                    if cfg.training.val_batches is not None and i>=cfg.training.val_batches:break
                    loss=model.cell_loss(b['history'].to(dev),b['target'].to(dev),b['condition'].to(dev))
                    vt+=float(loss)*len(b['target']);vc+=len(b['target'])
        if not vc or not np.isfinite(vt):raise RuntimeError('Invalid validation loss')
        value=vt/vc;improved=value<best;best=min(best,value)
        metrics={'epoch':epoch,'step':step,'train_loss':total/count,'val_loss':value,'train_samples':count,'val_samples':vc}
        print(json.dumps(metrics),flush=True)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        save_checkpoint(out/'last.pt',model,opt,cfg,index,epoch,step,best,membership)
        if improved:save_checkpoint(out/'best.pt',model,opt,cfg,index,epoch,step,best,membership)
    return report

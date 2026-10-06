"""Held-out test generation: one-step metrics and autoregressive x-space rollout."""
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from mitocheck_trainer import make_index, make_model, sampler, seed_all, validate_contract, write_json


def image_rollout(sampling, history, conditions):
    """Every generated image is the next clean history input; no target feedback."""
    outputs=[]
    for cond in conditions:
        prediction,_=sampling.sample(history,cond)
        outputs.append(prediction)
        history=torch.cat([history[:,1:],prediction[:,None]],dim=1)
    return outputs


def measure(pred,target):
    d=(pred-target)/2  # metrics in [0,1] intensity units
    mse=float(d.square().mean());mae=float(d.abs().mean())
    return mse,mae


def run_evaluation(checkpoint,root,output,device='cuda',horizon=10,max_windows=None,save_grids=16):
    if horizon<1 or (max_windows is not None and max_windows<1):raise ValueError('Positive horizon/window limit required')
    out=Path(output).resolve()
    if (out/'metrics.json').exists():raise ValueError('Evaluation output already exists; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=OmegaConf.create(state['config']);cfg.dataset.root=str(root)
    index=make_index(cfg);validate_contract(state,index,cfg)
    dev=torch.device(device);torch.set_num_threads(cfg.cpu_threads);seed_all(cfg.seed)
    model=make_model(cfg,index.genes.shape[1]).to(dev);model.load_state_dict(state['model']);model.eval()
    sampling=sampler(model,cfg)
    sums={};count=0;grids=0;scored_tracks=set();by_gene={}
    def add(key,pred,target):
        a,b=measure(pred,target);v=sums.setdefault(key,[0.,0.,0]);v[0]+=a;v[1]+=b;v[2]+=1
        return a,b
    with torch.no_grad():
        for uid in sorted(index.fixed):
            if index.fixed[uid]!='test':continue
            for segment in index.segments[uid]:
                k=cfg.dataset.history
                for start in range(len(segment)-k):
                    if max_windows is not None and count>=max_windows:break
                    source=segment[start+k-1];future=segment[start+k:start+k+horizon]
                    history=torch.stack([index.image(r) for r in segment[start:start+k]])[None].to(dev)
                    conditions=[]
                    for j,r in enumerate(future):
                        previous=source if j==0 else future[j-1]
                        conditions.append(index.condition(previous,r['frame_id'],cfg.dataset.interval,cfg.dataset.offset)[None].to(dev))
                    predictions=image_rollout(sampling,history,conditions)
                    for j,(prediction,row) in enumerate(zip(predictions,future)):
                        target=index.image(row)[None].to(dev)
                        a,b=add(f'rollout_{j+1}',prediction,target)
                        if j==0:
                            add('one_step',prediction,target)
                            v=by_gene.setdefault(source['Gene'],[0.,0.,0]);v[0]+=a;v[1]+=b;v[2]+=1
                    if grids<save_grids:
                        top=[history[0,-1,0].cpu()]+[p[0,0].cpu() for p in predictions]
                        bottom=[history[0,-1,0].cpu()]+[index.image(r)[0] for r in future]
                        grid=torch.cat([torch.cat(top,dim=1),torch.cat(bottom,dim=1)],dim=0)
                        Image.fromarray(((grid.clamp(-1,1)+1)*127.5).numpy().astype(np.uint8)).save(out/f'rollout_{grids:03d}.png')
                        grids+=1
                    count+=1;scored_tracks.add(uid)
                if max_windows is not None and count>=max_windows:break
            if max_windows is not None and count>=max_windows:break
    if not count:raise ValueError('No test windows for requested history')
    def summary(v):
        mse=v[0]/v[2]
        return {'mse':mse,'mae':v[1]/v[2],'psnr':-10*np.log10(max(mse,1e-12)),'count':v[2]}
    metrics={'fixed_test_tracks':index.contract['fixed_track_counts']['test'],'scored_tracks':len(scored_tracks),'windows':count,
             'metrics':{key:summary(v) for key,v in sums.items()},'one_step_by_gene':{key:summary(v) for key,v in by_gene.items()},
             'contract':index.contract,'sampling':OmegaConf.to_container(cfg.sampler),'seed':cfg.seed,
             'horizon':horizon,'max_windows':max_windows,'intensity_range':[0,1],
             'note':'one_step equals rollout step 1; later rollout predictions use generated image history; gaps terminate segments',
             'checkpoint_epoch':state['epoch']}
    write_json(out/'metrics.json',metrics); print(json.dumps(metrics,indent=2));return metrics


@hydra.main(version_base=None,config_path='../config',config_name='mitocheck')
def main(cfg:DictConfig):
    if not cfg.checkpoint:raise ValueError('Set checkpoint=/path/to/best.pt')
    run_evaluation(cfg.checkpoint,cfg.dataset.root,cfg.output_dir,cfg.device,cfg.evaluation.horizon,cfg.evaluation.max_windows,cfg.evaluation.save_grids)

if __name__=='__main__':main()

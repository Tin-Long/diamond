import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mitocheck_data import TrackIndex, CellWindows
from models.diffusion import Denoiser, DenoiserConfig, InnerModelConfig, SigmaDistributionConfig, DiffusionSampler, DiffusionSamplerConfig


class CellTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        r = self.root / 'training_ready_v2'
        (r / 'embeddings').mkdir(parents=True)
        np.savez(r / 'embeddings/gene_emb.npz', keys=np.array(['G']), embeddings=np.ones((1, 200), dtype=np.float32))
        s = self.root / 'splits/gene_track_split_v1'
        s.mkdir(parents=True)
        c = sqlite3.connect(s / 'split_index.sqlite')
        c.execute('create table tracks(track_uid text, split text, pair_eligible int)')
        c.executemany('insert into tracks values(?,?,?)', [('a','train',1),('b','train',1),('c','test',1),('extra','train',0)])
        c.commit();c.close()
        c = sqlite3.connect(r / 'index.sqlite')
        c.execute('create table tracks(track_uid text,start_frame int,split text,retained_for_pairs int)')
        c.execute('create table conditions(Gene text,gene_key text,gene_index int)')
        c.execute("insert into conditions values('G','G',0)")
        c.execute('create table cells(cell_uid text,track_uid text,frame_id int,image_path text,Gene text,gene_index int,split text,paired_eligible int,sample_type text)')
        for track, split, frames in [('a','train',[0,1,3,4]),('b','train',[2,3]),('c','test',[0,1,2]),('extra','train',[0,1])]:
            c.execute('insert into tracks values(?,?,?,?)', (track,frames[0],split,int(track!='extra')))
            for f in frames:
                name=f'{track}_{f}.tif'
                Image.fromarray(np.full((96,96),f,dtype=np.uint8)).save(r/name)
                c.execute('insert into cells values(?,?,?,?,?,?,?,?,?)',(name,track,f,name,'G',0,split,int(track!='extra'),'single'))
        c.commit();c.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_fixed_membership_and_gaps(self):
        index=TrackIndex(self.root,expected_train=2,expected_test=1)
        ds=CellWindows(index,'train',history=1,val_fraction=0)
        self.assertEqual(len(ds),3)
        self.assertEqual({x['track_uid'] for x in ds}, {'a','b'})
        for x in ds:
            self.assertEqual(x['target_frame']-x['source_frame'],1)
            self.assertEqual(tuple(x['history'].shape),(1,1,96,96))
            self.assertAlmostEqual(float(x['condition'][-3]),19+.5*x['source_frame'])
        self.assertEqual(len(CellWindows(index,'test',val_fraction=0)),2)
        self.assertEqual(index.audit['fixed_track_counts'],{'train':2,'test':1})

    def test_partition_and_missing_embedding(self):
        index=TrackIndex(self.root,expected_train=2,expected_test=1)
        tr=CellWindows(index,'train',val_fraction=.5)
        va=CellWindows(index,'val',val_fraction=.5)
        self.assertFalse(set(tr.track_ids)&set(va.track_ids))
        self.assertEqual(set(tr.track_ids)|set(va.track_ids),{'a','b'})
        self.assertEqual(va.track_ids,CellWindows(index,'val',val_fraction=.5).track_ids)
        with self.assertRaises(ValueError): TrackIndex(self.root,expected_train=99,expected_test=1)

    def test_control_uses_dataset_gene_key(self):
        r=self.root/'training_ready_v2'
        np.savez(r/'embeddings/gene_emb.npz',keys=np.array(['NO_TARGET']),embeddings=np.zeros((1,200),dtype=np.float32))
        with sqlite3.connect(r/'index.sqlite') as c:
            c.execute("update cells set Gene='NEGATIVE_CONTROL'")
            c.execute("update conditions set Gene='NEGATIVE_CONTROL',gene_key='NO_TARGET'")
        index=TrackIndex(self.root,expected_train=2,expected_test=1)
        self.assertEqual(len(CellWindows(index,'test',val_fraction=0)),2)

    def test_history_never_crosses_gaps(self):
        index=TrackIndex(self.root,expected_train=2,expected_test=1)
        self.assertEqual(len(CellWindows(index,'train',history=2,val_fraction=0)),0)
        self.assertEqual(len(CellWindows(index,'test',history=2,val_fraction=0)),1)

    def test_missing_embedding_and_wrong_image(self):
        r=self.root/'training_ready_v2'
        with sqlite3.connect(r/'index.sqlite') as c:c.execute("update cells set gene_index=NULL where track_uid='a'")
        with self.assertRaises(ValueError):TrackIndex(self.root,expected_train=2,expected_test=1)
        with sqlite3.connect(r/'index.sqlite') as c:c.execute("update cells set gene_index=0")
        Image.fromarray(np.zeros((20,20),dtype=np.uint8)).save(r/'c_1.tif')
        index=TrackIndex(self.root,expected_train=2,expected_test=1)
        with self.assertRaises(ValueError):CellWindows(index,'test')[0]

    def test_rollout_feeds_generated_images(self):
        from evaluate_mitocheck import image_rollout
        class Fake:
            def __init__(self):self.calls=[]
            def sample(self,h,c):
                self.calls.append((h.clone(),c.clone()))
                return h[:,-1]+1,[]
        fake=Fake();h=torch.zeros(1,2,1,4,4)
        conditions=[torch.tensor([[19.,19.5]]),torch.tensor([[19.5,20.]])]
        output=image_rollout(fake,h,conditions)
        self.assertTrue(torch.equal(fake.calls[1][0][:,-1],output[0]))
        self.assertEqual(float(fake.calls[1][1][0,-1]),20.)

    def test_train_resume_and_heldout_evaluation(self):
        from omegaconf import OmegaConf
        from mitocheck_trainer import run,validate_contract
        from evaluate_mitocheck import run_evaluation
        cfg=OmegaConf.load(Path(__file__).resolve().parents[1]/'config/mitocheck.yaml')
        cfg.output_dir=str(self.root/'run');cfg.dataset.root=str(self.root)
        cfg.dataset.expected_train=2;cfg.dataset.expected_test=1;cfg.dataset.val_fraction=.5
        cfg.device='cpu';cfg.cpu_threads=1;cfg.training.epochs=1;cfg.training.workers=0
        cfg.training.max_batches=1;cfg.training.val_batches=1;cfg.training.batch_size=1
        cfg.model.cond_channels=32;cfg.model.depths=[1,1];cfg.model.channels=[16,16];cfg.model.attn_depths=[0,0]
        run(cfg)
        checkpoint=self.root/'run/last.pt'
        state=torch.load(checkpoint,weights_only=False)
        self.assertEqual(state['step'],1);self.assertEqual(state['epoch'],0)
        self.assertEqual(state['membership']['test'],['c'])
        cfg.resume=str(checkpoint);cfg.training.epochs=2;run(cfg)
        state=torch.load(checkpoint,weights_only=False)
        self.assertEqual(state['step'],2);self.assertEqual(state['epoch'],1)
        result=run_evaluation(checkpoint,self.root,self.root/'eval','cpu',horizon=2,max_windows=1,save_grids=1)
        self.assertEqual(result['scored_tracks'],1);self.assertEqual(result['metrics']['rollout_2']['count'],1)
        self.assertTrue((self.root/'eval/rollout_000.png').exists())
        cfg.dataset.offset=0
        with self.assertRaises(ValueError):validate_contract(state,TrackIndex(self.root,2,1),cfg)

    def test_loss_sampling_and_atari(self):
        torch.set_num_threads(1)
        cfg=InnerModelConfig(img_channels=1,num_steps_conditioning=1,cond_channels=32,depths=[1,1],channels=[16,16],attn_depths=[0,0],cell_condition_dim=203)
        model=Denoiser(DenoiserConfig(cfg,.5,.3))
        model.setup_training(SigmaDistributionConfig(-.4,1.2,.002,20))
        index=TrackIndex(self.root,expected_train=2,expected_test=1)
        item=CellWindows(index,'train',val_fraction=0)[0]
        history=item['history'][None]; target=item['target'][None]; cond=item['condition'][None]
        loss=model.cell_loss(history,target,cond)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertGreater(float(model.inner_model.conv_out.weight.grad.abs().sum()),0)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3);opt.step();opt.zero_grad()
        model.cell_loss(history,target,cond).backward()
        opt.step();opt.zero_grad()
        model.cell_loss(history,target,cond).backward()
        self.assertGreater(float(model.inner_model.act_emb[0].weight.grad.abs().sum()),0)
        sampler=DiffusionSampler(model,DiffusionSamplerConfig(3))
        image,_=sampler.sample(history,cond)
        self.assertEqual(image.shape,target.shape);self.assertTrue(torch.isfinite(image).all())
        discrete=InnerModelConfig(img_channels=3,num_steps_conditioning=4,cond_channels=32,depths=[1,1],channels=[16,16],attn_depths=[0,0],num_actions=2)
        legacy=Denoiser(DenoiserConfig(discrete,.5,.3))
        image,_=DiffusionSampler(legacy,DiffusionSamplerConfig(3)).sample(torch.zeros(1,4,3,16,16),torch.zeros(1,4,dtype=torch.long))
        self.assertEqual(tuple(image.shape),(1,3,16,16))

if __name__=='__main__':unittest.main()

"""Read-only fixed-track MitoCheck adapter; current TIFF paths come from v2."""
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_db(path):
    c = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    return c


class TrackIndex:
    def __init__(self, root, expected_train=6571, expected_test=1642):
        base = Path(root).expanduser().resolve()
        self.root = base if base.name == 'training_ready_v2' else base / 'training_ready_v2'
        split_db = self.root.parent / 'splits/gene_track_split_v1/split_index.sqlite'
        with read_db(split_db) as c:
            fixed = {r['track_uid']: r['split'] for r in c.execute("select track_uid,split from tracks where pair_eligible=1")}
        if set(fixed.values()) - {'train', 'test'}:
            raise ValueError('Invalid fixed track split')
        counts = {s: sum(x == s for x in fixed.values()) for s in ('train', 'test')}
        if counts != {'train': expected_train, 'test': expected_test}:
            raise ValueError(f'Fixed pair track counts {counts}; expected {expected_train}/{expected_test}')
        emb_path = self.root / 'embeddings/gene_emb.npz'
        with np.load(emb_path, allow_pickle=False) as e:
            self.genes = e['embeddings'].astype(np.float32)
            keys = [str(x) for x in e['keys']]
        if self.genes.ndim != 2 or not np.isfinite(self.genes).all():
            raise ValueError('Invalid gene embeddings')
        self.tracks = defaultdict(list)
        self.births = {}
        db_path = self.root / 'index.sqlite'
        with read_db(db_path) as c:
            gene_keys = {}
            for r in c.execute('select Gene,gene_key,gene_index from conditions'):
                key=(r['Gene'],r['gene_index'])
                if key in gene_keys and gene_keys[key] != r['gene_key']:
                    raise ValueError('Inconsistent condition identity')
                gene_keys[key]=r['gene_key']
            for r in c.execute("select track_uid,start_frame,split,retained_for_pairs from tracks where split in ('train','test')"):
                if r['track_uid'] in fixed:
                    if r['split'] != fixed[r['track_uid']] or not r['retained_for_pairs']:
                        raise ValueError('Fixed track membership differs from current index')
                    self.births[r['track_uid']] = int(r['start_frame'])
            for row in c.execute("select cell_uid,track_uid,frame_id,image_path,Gene,gene_index,split from cells where paired_eligible=1 and sample_type='single' order by track_uid,frame_id"):
                r = dict(row); uid = r['track_uid']
                if uid not in fixed or r['split'] != fixed[uid]:
                    raise ValueError(f'Eligible cell outside fixed membership: {uid}')
                gi = r['gene_index']
                if gi is None or not 0 <= gi < len(keys) or keys[gi] != gene_keys.get((r['Gene'],gi)):
                    raise ValueError(f'Missing/mismatched Gene2Vec for {uid}')
                path = Path(r['image_path'])
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Image paths must be relative to current v2 root')
                self.tracks[uid].append(r)
        if set(self.tracks) != set(fixed) or set(self.births) != set(fixed):
            raise ValueError('Current eligible tracks do not match fixed membership')
        self.fixed = fixed
        self.segments = {}
        gaps = 0
        for uid, rows in self.tracks.items():
            segments = [[]]
            seen = set()
            for row in rows:
                if row['cell_uid'] in seen or (segments[-1] and row['frame_id'] <= segments[-1][-1]['frame_id']):
                    raise ValueError(f'Duplicate/non-monotone track {uid}')
                seen.add(row['cell_uid'])
                if segments[-1] and row['frame_id'] != segments[-1][-1]['frame_id'] + 1:
                    segments.append([]); gaps += 1
                segments[-1].append(row)
            self.segments[uid] = segments
        self.contract = {'split_sha256': digest(split_db), 'index_sha256': digest(db_path),
                         'gene_sha256': digest(emb_path), 'gene_dim': self.genes.shape[1],
                         'fixed_track_counts': counts, 'normalization': 'uint8/255*2-1',
                         'sample_type': 'single', 'gaps': 'split into contiguous segments'}
        self.audit = {**self.contract, 'gap_boundaries': gaps,
                      'eligible_cell_counts': {s: sum(len(self.tracks[u]) for u in fixed if fixed[u] == s) for s in counts}}

    def image(self, row):
        path = self.root / row['image_path']
        with Image.open(path) as im:
            a = np.asarray(im)
        if a.shape != (96, 96) or a.dtype != np.uint8:
            raise ValueError(f'Expected 96x96 uint8 grayscale TIFF: {path}, got {a.shape}/{a.dtype}')
        return torch.from_numpy(a.astype(np.float32)).unsqueeze(0).div(255).mul(2).sub(1)

    def condition(self, row, target_frame, interval=.5, offset=19.):
        uid = row['track_uid']
        time = [offset + row['frame_id'] * interval, offset + target_frame * interval,
                (row['frame_id'] - self.births[uid]) * interval]
        return torch.from_numpy(np.concatenate([self.genes[row['gene_index']], np.asarray(time, dtype=np.float32)]))


class CellWindows(Dataset):
    def __init__(self, index, split, history=1, val_fraction=.05, seed=0, interval=.5, offset=19.):
        if split not in ('train', 'val', 'test') or history < 1 or not 0 <= val_fraction < 1:
            raise ValueError('Invalid split/history/validation fraction')
        self.index = index; self.history = history; self.interval = interval; self.offset = offset
        groups = defaultdict(list)
        for uid, fixed_split in index.fixed.items():
            if fixed_split == 'train': groups[index.tracks[uid][0]['Gene']].append(uid)
        heldout = set()
        for members in groups.values():
            ordered = sorted(members, key=lambda u: hashlib.sha256(f'{seed}:{u}'.encode()).hexdigest())
            n = min(len(ordered) - 1, max(1, round(len(ordered) * val_fraction))) if val_fraction and len(ordered) > 1 else 0
            heldout.update(ordered[:n])
        self.track_ids = sorted(u for u, s in index.fixed.items() if
            (s == 'test' if split == 'test' else s == 'train' and ((u in heldout) if split == 'val' else (u not in heldout))))
        self.windows = []
        for uid in self.track_ids:
            for segment in index.segments[uid]:
                for start in range(len(segment) - history):
                    self.windows.append(segment[start:start + history + 1])

    def __len__(self): return len(self.windows)

    def __getitem__(self, i):
        rows = self.windows[i]; source = rows[-2]; target = rows[-1]
        return {'history': torch.stack([self.index.image(r) for r in rows[:-1]]),
                'target': self.index.image(target),
                'condition': self.index.condition(source, target['frame_id'], self.interval, self.offset),
                'track_uid': source['track_uid'], 'source_uid': source['cell_uid'], 'target_uid': target['cell_uid'],
                'source_frame': source['frame_id'], 'target_frame': target['frame_id'], 'gene': source['Gene']}

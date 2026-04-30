"""
Align THINGS-EEG2 events.tsv rows with their stimulus images.

The dataset stores images as two zips under stimuli/ and a metadata .npy
mapping per-trial `tot_img_number` to a (concept, filename) pair:

    stimuli/training_images.zip   (16,540 train images)
    stimuli/test_images.zip       (200 test images)
    stimuli/image_metadata.npy    (lookup tables)

Each row of *_task-train_*_events.tsv / *_task-test_*_events.tsv has:
    trial_type ∈ {image, target, rest_marker}
    tot_img_number  - 1-based index into train_img_files / test_img_files
                      (n/a on target trials)
    category_name   - matches the concept directory inside the zip

Usage:
    aligner = StimulusAligner(root='/path/to/nm000232')
    img = aligner.image_for_event(row)            # returns PIL.Image
    paths = aligner.path_for_events(events_df)    # vectorised: list of Path

The first call extracts the zips into stimuli/_extracted/ (idempotent).
Subsequent calls just resolve paths.
"""
from __future__ import annotations
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


class StimulusAligner:
    def __init__(self, root: str | Path, extracted_dir: Optional[Path] = None):
        self.root = Path(root)
        self.stim = self.root / 'stimuli'
        # Default extraction cache lives under stimuli/ so the dataset
        # stays self-contained, but callers can redirect to a fast scratch.
        self.extracted = Path(extracted_dir) if extracted_dir else self.stim / '_extracted'

        meta = np.load(self.stim / 'image_metadata.npy', allow_pickle=True).item()
        # 1-based tot_img_number → concept directory + filename
        self._train_concepts = list(meta['train_img_concepts'])
        self._train_files = list(meta['train_img_files'])
        self._test_concepts = list(meta['test_img_concepts'])
        self._test_files = list(meta['test_img_files'])

    # ---- extraction ----

    def ensure_extracted(self, kind: str) -> Path:
        """Extract the zip for `kind` ∈ {'train', 'test'} if not already done.

        Idempotent and safe to call from multiple processes (uses a marker
        file so we don't pay the unzip cost twice).
        """
        zip_name = 'training_images.zip' if kind == 'train' else 'test_images.zip'
        out_root = self.extracted / ('training_images' if kind == 'train' else 'test_images')
        marker = self.extracted / f'.{kind}_extracted'
        if marker.exists() and out_root.exists():
            return out_root
        out_root.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.stim / zip_name) as z:
            z.extractall(self.extracted)
        marker.touch()
        return out_root

    # ---- single-row resolution ----

    def path_for_event(self, row) -> Optional[Path]:
        """Return the image path for one events.tsv row, or None for non-image trials."""
        if str(row.get('trial_type', '')) != 'image':
            return None
        ti = row.get('tot_img_number')
        if pd.isna(ti) or ti == 'n/a':
            return None
        idx = int(ti) - 1
        # Heuristic: train uses 1..16540, test uses 1..200; the events.tsv
        # naming (task-train vs task-test) is the truth source. We accept
        # an explicit `kind` hint for vectorised path resolution below.
        if 0 <= idx < len(self._test_files) and len(self._test_files) >= idx + 1 and idx < 200:
            # ambiguous range; fall through to use the events filename hint
            pass
        # Without a kind hint, infer from the magnitude of tot_img_number:
        if idx < 200 and self._test_concepts and idx < len(self._test_concepts):
            kind = 'test'
        else:
            kind = 'train'
        return self._resolve(idx, kind)

    def image_for_event(self, row, mode: str = 'PIL'):
        """Return the stimulus image. mode ∈ {'PIL', 'bytes', 'path'}."""
        p = self.path_for_event(row)
        if p is None:
            return None
        if mode == 'path':
            return p
        if mode == 'bytes':
            return p.read_bytes()
        from PIL import Image
        return Image.open(p)

    # ---- vectorised resolution (the path you actually want for training) ----

    def paths_for_events(self, events: pd.DataFrame, kind: str) -> list[Optional[Path]]:
        """Given a full events.tsv DataFrame and the run kind ('train' or 'test'),
        return one Path per row (None for non-image trials).

        Pass `kind` from the BIDS task name; this avoids the magnitude heuristic.
        """
        self.ensure_extracted(kind)
        out: list[Optional[Path]] = []
        for _, row in events.iterrows():
            if str(row.get('trial_type', '')) != 'image':
                out.append(None); continue
            ti = row.get('tot_img_number')
            if pd.isna(ti) or ti == 'n/a':
                out.append(None); continue
            out.append(self._resolve(int(ti) - 1, kind))
        return out

    # ---- internals ----

    def _resolve(self, idx0: int, kind: str) -> Path:
        if kind == 'train':
            concept = self._train_concepts[idx0]
            fname = self._train_files[idx0]
            sub = 'training_images'
        else:
            concept = self._test_concepts[idx0]
            fname = self._test_files[idx0]
            sub = 'test_images'
        out_root = self.ensure_extracted(kind)
        return out_root / concept / fname


def demo(root: str = '/data/tau/iceberg_1/titanic_1/datasets/bids/nm000232',
         subject: str = '01', session: str = '01') -> None:
    """Print a short alignment proof for the first few image events."""
    root_p = Path(root)
    aligner = StimulusAligner(root_p)
    for kind, suffix in [('test', 'test'), ('train', 'train_run-01')]:
        ev = root_p / f'sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{suffix}_events.tsv'
        df = pd.read_csv(ev, sep='\t').head(5)
        print(f'\n=== {kind} ({ev.name}) ===')
        paths = aligner.paths_for_events(df, kind=kind)
        for (_, r), p in zip(df.iterrows(), paths):
            ok = (p is not None and p.exists())
            print(f'  trial_type={r.trial_type:8s}  tot_img_number={r.tot_img_number}  '
                  f'category_name={r.category_name}  →  {p}  (exists={ok})')


if __name__ == '__main__':
    demo()

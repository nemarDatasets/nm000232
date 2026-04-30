"""
End-to-end smoke test: load nm000232 with braindecode.datasets.BIDSDataset,
extract image-locked epochs around each `image` event, and resolve every
epoch's stimulus image via the StimulusAligner.

Demonstrates that the EEG ↔ image alignment is intact (no missing files,
ordering correct).
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mne
from braindecode.datasets import BIDSDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_stimuli import StimulusAligner


ROOT = Path('/data/tau/iceberg_1/titanic_1/datasets/bids/nm000232')


def run(subject='01', session='01', task='test'):
    print(f'>> loading sub-{subject} ses-{session} task-{task} via BIDSDataset')
    ds = BIDSDataset(
        root=str(ROOT),
        subjects=[subject],
        sessions=[session],
        tasks=[task],
        datatypes=['eeg'],
        extensions=['.bdf'],
    )
    print(f'   BIDSDataset: {len(ds.datasets)} runs found')
    aligner = StimulusAligner(ROOT)

    total_image, total_resolved, total_missing = 0, 0, 0
    for sub_ds in ds.datasets:
        raw: mne.io.BaseRaw = sub_ds.raw
        sfreq = raw.info['sfreq']
        # Find the matching events.tsv. mne_bids stores BIDSPath in
        # sub_ds.target_name / sub_ds.description; simpler: derive from
        # raw.filenames[0].
        bdf = Path(raw.filenames[0])
        ev_tsv = bdf.with_name(bdf.name.replace('_eeg.bdf', '_events.tsv'))
        ev = pd.read_csv(ev_tsv, sep='\t')

        n_image = (ev.trial_type == 'image').sum()
        paths = aligner.paths_for_events(ev, kind=task)
        resolved = sum(1 for p in paths if p is not None and p.exists())
        missing = n_image - resolved

        # Build epochs around image events to prove EEG-side alignment too.
        # Use the first 5 image events to keep this light.
        image_rows = ev[ev.trial_type == 'image'].head(5)
        events = np.array([
            [int(round(r.onset * sfreq)), 0, 1] for _, r in image_rows.iterrows()
        ], dtype=int)
        epochs = mne.Epochs(
            raw, events, event_id={'image': 1},
            tmin=-0.1, tmax=0.5, baseline=(-0.1, 0.0),
            preload=True, verbose='ERROR',
        )
        x = epochs.get_data()
        print(f'   run={bdf.name}: events={len(ev)}  image-events={n_image}  '
              f'resolved={resolved}  missing={missing}  '
              f'epoch_shape(first5)={x.shape}')
        # Show a paired EEG + image for the first 3 events
        for i, (_, r) in enumerate(image_rows.iterrows()):
            p = aligner.path_for_event(r) or aligner._resolve(int(r.tot_img_number) - 1, task)
            print(f'      [{i}] onset={r.onset:.3f}s  tot_img={int(r.tot_img_number)}  '
                  f'cat={r.category_name}  eeg=(c={x.shape[1]}, t={x.shape[2]})  img={p.name}')

        total_image += n_image
        total_resolved += resolved
        total_missing += missing

    print(f'\n== summary ==')
    print(f'   image events: {total_image}')
    print(f'   resolved    : {total_resolved}')
    print(f'   missing     : {total_missing}')
    if total_missing == 0:
        print('   ✓ EEG ↔ stimulus alignment intact')
    else:
        print(f'   ✗ {total_missing} unresolved — investigate')


if __name__ == '__main__':
    run(task='test')
    print()
    run(task='train')

"""
Restructure stimuli/ to follow BEP044 (BIDS-stim, PR #2022) as closely as possible.

Original layout (kept as-is for backup / interoperability):
    stimuli/
        training_images.zip   (655 MB, 16,540 images grouped by concept)
        test_images.zip       (8 MB, 200 images grouped by concept)
        image_metadata.npy    (lookup: tot_img_number -> concept, filename)
        LICENSE.txt           (THINGS database license)

New BEP044 layout (added side-by-side):
    stimuli/
        stimuli.tsv                    REQUIRED marker file (BEP044)
        stimuli.json                   column descriptions
        training_images/<concept>/stim-train<NNNNN>_image.jpg
        test_images/<concept>/stim-test<NNN>_image.jpg

Naming:
  - stimulus_id "stim-train<NNNNN>" with NNNNN = 1-based `tot_img_number`,
    zero-padded to 5 digits (max 16540).
  - stimulus_id "stim-test<NNN>"   with NNN  = 1-based `tot_img_number`,
    zero-padded to 3 digits (max 200).
  - Files renamed to "stim-<label>_image.jpg"; concept subdirs preserved.

This script is idempotent: rerunning it skips already-extracted images and
overwrites stimuli.tsv / stimuli.json from scratch.
"""
from __future__ import annotations
import csv
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path('/data/tau/iceberg_1/titanic_1/datasets/bids/nm000232')
STIM = ROOT / 'stimuli'

LICENSE_NOTE = (
    "Original image copyright belongs to image owners; THINGS database images "
    "are available for research purposes only under fair use (see stimuli/LICENSE.txt). "
    "Cite Hebart et al. 2019 (doi:10.1101/545954)."
)
THINGS_URL = "https://things-initiative.org/"


def load_metadata():
    md = np.load(STIM / 'image_metadata.npy', allow_pickle=True).item()
    return {
        'train_concepts': list(md['train_img_concepts']),
        'train_files':    list(md['train_img_files']),
        'test_concepts':  list(md['test_img_concepts']),
        'test_files':     list(md['test_img_files']),
    }


def extract_zip(zip_path: Path, dest_root: Path) -> None:
    """Extract `zip_path` into `dest_root` (idempotent: skip if marker exists)."""
    marker = dest_root / '.extracted'
    if marker.exists():
        return
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_root.parent)  # zip already contains top-level dir
    marker.touch()


def stim_label(kind: str, idx1: int) -> str:
    if kind == 'train':
        return f'train{idx1:05d}'   # train00001..train16540
    return f'test{idx1:03d}'        # test001..test200


def rename_to_bids(kind: str, concepts: list[str], files: list[str]) -> list[dict]:
    """For each (concept, original_filename) pair, produce a BIDS-named copy at
    the expected path and return one stimuli.tsv row."""
    sub = 'training_images' if kind == 'train' else 'test_images'
    base = STIM / sub
    rows = []
    for i, (concept, fname) in enumerate(zip(concepts, files), start=1):
        src = base / concept / fname
        ext = src.suffix.lower().lstrip('.')      # jpg, png, ...
        label = stim_label(kind, i)
        new_name = f'stim-{label}_image.{ext}'
        dst = base / concept / new_name
        if not dst.exists():
            if not src.exists():
                raise FileNotFoundError(
                    f'Expected source image missing: {src}\n'
                    f'  (re-run extract step first.)'
                )
            shutil.copy2(src, dst)                # keep original next to BIDS-named copy
        rows.append({
            'stimulus_id': f'stim-{label}',
            'type': 'image',
            'license': 'n/a',
            'copyright': 'Original image owners (THINGS database, Hebart et al. 2019)',
            'description': (
                f'THINGS-EEG2 {kind} stimulus tot_img_number={i}; '
                f'concept "{concept}"; original filename "{fname}".'
            ),
            'URL': THINGS_URL,
            'filename': f'{sub}/{concept}/{new_name}',
            'present': 'true',
        })
    return rows


def write_stimuli_tsv(rows: list[dict]) -> None:
    cols = ['stimulus_id', 'type', 'license', 'copyright', 'description',
            'URL', 'filename', 'present']
    out = STIM / 'stimuli.tsv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'  wrote {out} ({len(rows)} rows)')


def write_stimuli_json() -> None:
    sidecar = {
        'stimulus_id': {
            'Description': 'Unique identifier for a specific stimulus. '
                           'Maps to events.tsv "tot_img_number": for "stim-trainNNNNN" '
                           'or "stim-testNNN", strip the prefix and remove leading '
                           'zeros to recover the 1-based image index used in events.tsv.',
        },
        'type': {
            'Description': 'Stimulus modality. Refers to the BIDS suffix of the '
                           'stimulus file.',
            'Levels': {'image': 'A static visual stimulus (jpg).'},
        },
        'license': {
            'Description': 'License of the stimulus file. Set to "n/a" because '
                           'individual THINGS images are released under fair-use, '
                           'research-only terms (see stimuli/LICENSE.txt).',
        },
        'copyright': {
            'Description': 'Copyright holder. THINGS images belong to their '
                           'original owners; published in Hebart et al. 2019.',
        },
        'description': {
            'Description': 'Free-form description; encodes the original concept '
                           'directory and original filename for traceability.',
        },
        'URL': {
            'Description': 'Project URL for the THINGS initiative.',
        },
        'filename': {
            'Description': 'Path to the stimulus file relative to the stimuli/ '
                           'directory.',
        },
        'present': {
            'Description': 'Whether the file is bundled with the dataset.',
            'Levels': {'true': 'File is present.', 'false': 'File is not present.'},
        },
        'BEP044Adherence': {
            'Description': 'This dataset follows BEP044 (BIDS-stim) as proposed '
                           'in https://github.com/bids-standard/bids-specification/pull/2022. '
                           'events.tsv currently uses tot_img_number rather than the '
                           'spec-suggested stim_id column; tot_img_number maps to '
                           'stimulus_id by the convention encoded above.',
        },
    }
    out = STIM / 'stimuli.json'
    out.write_text(json.dumps(sidecar, indent=2) + '\n')
    print(f'  wrote {out}')


def main():
    md = load_metadata()
    print('Step 1: extract zips (idempotent)')
    extract_zip(STIM / 'training_images.zip', STIM / 'training_images')
    extract_zip(STIM / 'test_images.zip',     STIM / 'test_images')

    print('Step 2: write BIDS-named copies')
    train_rows = rename_to_bids('train', md['train_concepts'], md['train_files'])
    test_rows  = rename_to_bids('test',  md['test_concepts'],  md['test_files'])
    print(f'  train rows: {len(train_rows)}')
    print(f'  test  rows: {len(test_rows)}')

    print('Step 3: write stimuli.tsv + stimuli.json')
    # train first, then test, both already in tot_img_number order.
    write_stimuli_tsv(train_rows + test_rows)
    write_stimuli_json()

    # spot-check: a couple of resolved paths
    print('\nSpot-check:')
    for kind, idx in [('train', 11742), ('train', 1), ('test', 154), ('test', 200)]:
        if kind == 'train':
            row = train_rows[idx - 1]
        else:
            row = test_rows[idx - 1]
        p = STIM / row['filename']
        print(f'  {row["stimulus_id"]} → {row["filename"]}  exists={p.exists()}')


if __name__ == '__main__':
    main()

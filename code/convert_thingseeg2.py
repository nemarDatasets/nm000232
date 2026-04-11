#!/usr/bin/env python3
from __future__ import annotations

"""Convert THINGS-EEG2 (Gifford et al. 2022) to BIDS-EEG.

A 10-subject EEG dataset for modelling human visual object recognition,
acquired on a BrainVision actiCHamp system at 1000 Hz. The source files
record 63 EEG channels (one electrode served as the online reference and
is not stored). Each subject participated in 4 sessions with three task
types per session:

  - task-train (5 runs):  rapid serial visual presentation (RSVP) of 16,540
    distinct training images (~3,360 trials per run × 5 runs per session).
  - task-test (1 run):    RSVP of 200 test images, repeated ~80 times across
    the four sessions (~4,080 trials per session).
  - task-rest (2 runs):   2-minute resting state recordings, one before and
    one after the main experiment.

Stimulus details (from the paper and behavioural .mat files):
  - Image duration: 100 ms; SOA: 200 ms (5 Hz RSVP).
  - Each image presentation triggered a TTL pulse on the BrainVision
    recording. The trigger codes (1-99) cycle and do NOT identify the image
    uniquely. The actual image label has to be looked up in the per-run
    behavioural .mat file.
  - Target trials (random catch trials, ~6 per block) are marked with
    `tot_img_number == 0` in the behavioural files. We label them
    `trial_type == 'target'` in the BIDS events.tsv.

The original Source EEG download (10 subject zips) is bundled into BIDS as:

  /sourcedata/        — original BrainVision .eeg/.vhdr/.vmrk + behavioral .mat
  /derivatives/preprocessed_eeg/   — Gifford's preprocessed epochs (64-channel)
  /derivatives/resting_state/      — Gifford's preprocessed resting state
  /stimuli/           — image set + image_metadata.npy

Reference:
    Gifford, A.T., Dwivedi, K., Roig, G., & Cichy, R.M. (2022). A large and
    rich EEG dataset for modeling human visual object recognition. NeuroImage,
    264, 119754. https://doi.org/10.1016/j.neuroimage.2022.119754
    OSF: https://osf.io/3jk45/

Usage:
    python convert_thingseeg2.py --input /tmp/thingseeg2 --output /tmp/thingseeg2_bids
    python convert_thingseeg2.py --input /tmp/thingseeg2 --output /tmp/thingseeg2_bids \
        --max-subjects 1   # for testing
    python convert_thingseeg2.py --input /tmp/thingseeg2 --output /tmp/thingseeg2_bids \
        --skip-derivatives --skip-stimuli   # smaller test
"""

import argparse
import json
import logging
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import mne
import mne_bids
import numpy as np
import pandas as pd
from scipy.io import loadmat

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

EXPECTED_N_SUBJECTS = 10
N_SESSIONS = 4
N_TRAIN_PARTS = 5  # 5 train runs per session
N_REST_RUNS = 2  # rest1 (before) + rest2 (after)

# Authors from the published paper
AUTHORS = [
    "Alessandro T. Gifford",
    "Kshitij Dwivedi",
    "Gemma Roig",
    "Radoslaw M. Cichy",
]

# --------------------------------------------------------------------------
# Behavioral .mat extraction
# --------------------------------------------------------------------------


def _scalar(field: Any, default: Any = None) -> Any:
    """Recursively unwrap scipy.io scalar/array fields."""
    arr = field
    while hasattr(arr, "__len__") and not isinstance(arr, str):
        if len(arr) == 0:
            return default
        arr = arr[0]
    return arr


def parse_behavioral_mat(mat_path: Path) -> dict:
    """Parse a behavioral .mat file from THINGS-EEG2 source data.

    Returns a dict with:
      - subject_age: int
      - subject_sex: 'm' or 'f'
      - paradigm: dict of paradigm-level constants
      - trials: pd.DataFrame with one row per trial
    """
    beh = loadmat(str(mat_path))
    data = beh["data"][0][0]

    # Subject info
    sub_info = data["subject_info"]
    subject_age = int(_scalar(sub_info["age"]))
    subject_sex = str(_scalar(sub_info["sex"]))

    # Paradigm constants
    para = data["paradigm_info"]
    paradigm = {}
    for fname in para.dtype.names:
        val = _scalar(para[fname])
        if isinstance(val, (int, float, np.integer, np.floating)):
            paradigm[fname] = float(val)
        elif isinstance(val, str):
            paradigm[fname] = val

    # Per-trial info
    images = data["images"]  # shape (1, n_trials)
    n_trials = images.shape[1]

    rows = []
    for i in range(n_trials):
        t = images[0][i]
        # category_name is stored as a uint8 0 for target trials (not an
        # empty string), so fall back to "" only when the scalar is a proper
        # string.
        cat_raw = _scalar(t["category_name"], "")
        category_name = cat_raw if isinstance(cat_raw, str) else ""
        row = {
            "block": int(_scalar(t["block"], 0)),
            "sequence": int(_scalar(t["sequence"], 0)),
            "img_in_sequence": int(_scalar(t["img"], 0)),
            "img_type": str(_scalar(t["img_type"], "")),
            "img_category": int(_scalar(t["img_category"], 0)),
            "within_category_number": int(_scalar(t["within_category_number"], 0)),
            "tot_img_number": int(_scalar(t["tot_img_number"], 0)),
            "category_name": category_name,
            "trigger_code": int(_scalar(t["trigger_1"], 0)),
            "img_duration": float(_scalar(t["img_duration"], 0.1)),
            "soa": float(_scalar(t["SOA"], 0.2)),
        }
        rows.append(row)

    return {
        "subject_age": subject_age,
        "subject_sex": subject_sex,
        "paradigm": paradigm,
        "trials": pd.DataFrame(rows),
    }


# --------------------------------------------------------------------------
# Raw loading + event enrichment
# --------------------------------------------------------------------------


def load_brainvision(vhdr: Path) -> mne.io.BaseRaw:
    """Read a BrainVision file."""
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)
    # Apply standard 10-05 montage (the channel names match it)
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose=False)
    except Exception as exc:
        logger.debug("Montage failed: %s", exc)
    return raw


def build_events_df(
    raw: mne.io.BaseRaw,
    beh: dict | None,
    task: str,
) -> tuple[pd.DataFrame, mne.Annotations]:
    """Build the BIDS events DataFrame from a raw recording + behavioral data.

    For task-train and task-test, the behavioral .mat aligns 1:1 with the
    BrainVision stimulus annotations and provides per-trial image labels.
    For task-rest, the BV file has a couple of START/STOP markers we
    convert into a 'rest_start' / 'rest_end' annotation pair.
    """
    sfreq = raw.info["sfreq"]

    # Get BV stimulus annotations only (skip "New Segment" and other meta)
    stim_annots = [
        a for a in raw.annotations if str(a["description"]).startswith("Stimulus")
    ]

    rows: list[dict] = []
    descriptions: list[str] = []
    onsets: list[float] = []
    durations: list[float] = []

    if task in {"train", "test"} and beh is not None:
        trials = beh["trials"]
        if len(stim_annots) != len(trials):
            # >1% drift is suspicious — warn loudly so the user notices.
            drift = abs(len(stim_annots) - len(trials)) / max(len(trials), 1)
            level = logger.error if drift > 0.01 else logger.warning
            level(
                "Stimulus count mismatch: %d BV events vs %d beh trials "
                "(drift %.1f%%)",
                len(stim_annots),
                len(trials),
                drift * 100,
            )
            n = min(len(stim_annots), len(trials))
        else:
            n = len(stim_annots)

        for i in range(n):
            ann = stim_annots[i]
            tr = trials.iloc[i]
            onset = float(ann["onset"])
            duration = float(tr["img_duration"])
            is_target = tr["img_type"] == "target"
            trial_type = "target" if is_target else "image"
            # SOA of the very first trial in a run is undefined.
            soa_value: Any = "n/a" if i == 0 else float(tr["soa"])
            rows.append(
                {
                    "onset": onset,
                    "duration": duration,
                    "sample": int(round(onset * sfreq)),
                    "value": int(tr["trigger_code"]),
                    "trial_type": trial_type,
                    "tot_img_number": (
                        "n/a" if is_target else int(tr["tot_img_number"])
                    ),
                    "img_category": (
                        "n/a" if is_target else int(tr["img_category"])
                    ),
                    "category_name": (
                        "n/a" if is_target else (tr["category_name"] or "n/a")
                    ),
                    "block": int(tr["block"]),
                    "sequence": int(tr["sequence"]),
                    "img_in_sequence": int(tr["img_in_sequence"]),
                    "soa": soa_value,
                }
            )
            descriptions.append(trial_type)
            onsets.append(onset)
            durations.append(duration)
    else:
        # Resting state — keep whatever BV markers we have as generic events
        for ann in stim_annots:
            onset = float(ann["onset"])
            rows.append(
                {
                    "onset": onset,
                    "duration": 0.0,
                    "sample": int(round(onset * sfreq)),
                    "value": int(_extract_stim_code(str(ann["description"]))),
                    "trial_type": "rest_marker",
                    "tot_img_number": "n/a",
                    "img_category": "n/a",
                    "category_name": "n/a",
                    "block": "n/a",
                    "sequence": "n/a",
                    "img_in_sequence": "n/a",
                    "soa": "n/a",
                }
            )
            descriptions.append("rest_marker")
            onsets.append(onset)
            durations.append(0.0)

    df = pd.DataFrame(rows)
    annots = mne.Annotations(onsets, durations, descriptions, orig_time=None)
    return df, annots


def _extract_stim_code(desc: str) -> int:
    """Extract the integer trigger code from a BrainVision annotation."""
    m = re.search(r"S\s*(\d+)", desc)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------
# BIDS metadata writers
# --------------------------------------------------------------------------


def write_dataset_description(bids_root: Path) -> None:
    desc = {
        "Name": (
            "THINGS-EEG2: A large and rich EEG dataset for modeling human "
            "visual object recognition"
        ),
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "License": "CC-BY 4.0",
        "Authors": AUTHORS,
        "Acknowledgements": (
            "We thank all 10 participants and the Cichy and Roig labs at "
            "Freie Universität Berlin and Goethe University Frankfurt."
        ),
        "HowToAcknowledge": (
            "Please cite: Gifford, A.T., Dwivedi, K., Roig, G., & Cichy, R.M. "
            "(2022). A large and rich EEG dataset for modeling human visual "
            "object recognition. NeuroImage, 264, 119754. "
            "https://doi.org/10.1016/j.neuroimage.2022.119754"
        ),
        "Funding": [
            "German Research Council (DFG) grants CI 241/1-1, CI 241/3-1, CI 241/1-7",
            "European Research Council (ERC) Starting Grant ERC-2018-StG 803370",
            "Hessian Ministry of Higher Education, Research, Science and the Arts (HMWK)",
        ],
        "DatasetDOI": "doi:10.17605/OSF.IO/3JK45",
        "ReferencesAndLinks": [
            "https://osf.io/3jk45/",
            "https://doi.org/10.1016/j.neuroimage.2022.119754",
            "https://github.com/gifale95/eeg_encoding",
        ],
        "SourceDatasets": [{"URL": "https://osf.io/3jk45/"}],
        "GeneratedBy": [
            {
                "Name": "convert_thingseeg2.py (EEGDash)",
                "Description": (
                    "Converted from the THINGS-EEG2 Source EEG component "
                    "(original BrainVision files + behavioural .mat) to "
                    "BIDS-EEG. Source files preserved in /sourcedata/. "
                    "Authors' preprocessed data preserved in "
                    "/derivatives/preprocessed_eeg/ and "
                    "/derivatives/resting_state/. Image stimuli stored in "
                    "/stimuli/."
                ),
                "CodeURL": "https://github.com/bruaristimunha/EEGDash",
            }
        ],
        "HEDVersion": "8.2.0",
    }
    with open(bids_root / "dataset_description.json", "w") as f:
        json.dump(desc, f, indent=2)
        f.write("\n")


def write_readme(bids_root: Path) -> None:
    readme = """\
THINGS-EEG2: A large and rich EEG dataset for modeling human visual object recognition
========================================================================================

Overview
--------
EEG dataset of 10 subjects who viewed 16,540 distinct training images and 200
test images (each repeated ~80 times) using rapid serial visual presentation
(RSVP) at 5 Hz, recorded on a BrainVision actiCHamp system at 1000 Hz.
The source files store 63 EEG channels (the online reference electrode is
not stored). Stimuli are drawn from the THINGS database (Hebart et al. 2019).

Each subject completed 4 separate sessions; each session contained:
  - 5 training runs (~3,360 trials each) covering ~16,540 unique images
  - 1 test run (~4,080 trials) of 200 images repeated 20× per session
  - 2 resting-state runs (one before, one after the main experiment)

Total: ~32,540 training trials + ~16,000 test trials per subject across 4 sessions.

Recording setup
---------------
- Manufacturer: Brain Products (actiCHamp)
- 63 EEG channels (one electrode served as online reference and is not
  stored in the source files)
- 10-10 cap layout
- Sampling rate: 1000 Hz
- Online band-pass: 0.01-100 Hz
- Triggers recorded as BrainVision stimulus annotations (not as a
  dedicated stim channel)

Tasks (BIDS labels)
-------------------
- task-train: training run (RSVP of unique images)
- task-test:  test run (RSVP of repeated test images)
- task-rest:  resting state (eyes open, fixation cross)

Run numbering
-------------
- task-train: run-01..run-05 per session (5 training parts)
- task-test:  single run per session
- task-rest:  run-01 (before main task) and run-02 (after main task)

Events
------
events.tsv columns:
  onset, duration, sample, value, trial_type
  tot_img_number     - global image ID (1-16540 for train; 1-200 for test;
                       'n/a' for target catch trials)
  img_category       - integer category index
  category_name      - human-readable category, e.g. "01175_roller_coaster"
  block, sequence    - hierarchical position within the run
  img_in_sequence    - image position within its 20-image sequence
  soa                - actual stimulus onset asynchrony (~200 ms)

trial_type values:
  image  - normal training/test image presentation
  target - random catch trial (subject must press a button)
  rest_marker - resting-state start/end marker

Subject information
-------------------
participants.tsv contains age and sex (both extracted from the
behavioural .mat files in the source data).

Folder layout
-------------
/sub-XX/ses-YY/eeg/        - main BIDS data (BDF + sidecars)
/sourcedata/               - original BrainVision .eeg/.vhdr/.vmrk and
                             behavioural .mat files
/derivatives/preprocessed_eeg/   - authors' preprocessed train/test epochs
/derivatives/resting_state/      - authors' preprocessed resting state
/stimuli/                  - image set (training_images.zip, test_images.zip)
                             plus image_metadata.npy
/code/                     - this conversion script

Reference
---------
Gifford, A.T., Dwivedi, K., Roig, G., & Cichy, R.M. (2022). A large and rich
EEG dataset for modeling human visual object recognition. NeuroImage, 264,
119754. https://doi.org/10.1016/j.neuroimage.2022.119754

Code: https://github.com/gifale95/eeg_encoding
OSF:  https://osf.io/3jk45/
"""
    with open(bids_root / "README", "w") as f:
        f.write(readme)


def write_participants_json(bids_root: Path) -> None:
    desc = {
        "participant_id": {"Description": "Unique participant identifier (sub-XX)"},
        "age": {
            "Description": "Age of the participant at time of testing",
            "Units": "years",
        },
        "sex": {
            "Description": "Biological sex",
            "Levels": {"M": "Male", "F": "Female"},
        },
        "n_sessions": {"Description": "Number of EEG sessions completed"},
    }
    with open(bids_root / "participants.json", "w") as f:
        json.dump(desc, f, indent=2)
        f.write("\n")


def enrich_eeg_sidecar(
    sidecar_path: Path, task: str, run_idx: int | None = None
) -> None:
    """Add recommended fields to an EEG sidecar JSON."""
    if not sidecar_path.exists():
        return
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    # MNE-BIDS uses MiscChannelCount; BIDS spec also wants MISCChannelCount
    if "MiscChannelCount" in sidecar and "MISCChannelCount" not in sidecar:
        sidecar["MISCChannelCount"] = sidecar["MiscChannelCount"]

    if task == "train":
        task_desc = (
            "Rapid serial visual presentation (RSVP) of training images from "
            "the THINGS database. Each image was shown for 100 ms followed "
            "by 100 ms blank (200 ms SOA, 5 Hz). Subjects were instructed "
            "to maintain fixation and to press a button on random catch "
            "(target) trials."
        )
    elif task == "test":
        task_desc = (
            "Rapid serial visual presentation (RSVP) of 200 test images from "
            "the THINGS database. Each image was shown ~20 times per session. "
            "100 ms duration, 200 ms SOA, 5 Hz. Subjects fixated and "
            "responded to random catch (target) trials."
        )
    else:
        task_desc = (
            "Resting state EEG with central fixation cross. Recorded once "
            "before and once after the main experiment within each session."
        )

    sidecar.update(
        {
            "TaskName": task,
            "TaskDescription": task_desc,
            "Instructions": (
                "Maintain central fixation. On training/test runs, press the "
                "button as quickly as possible whenever a catch (target) "
                "image appears."
            ),
            "InstitutionName": "Freie Universität Berlin",
            "InstitutionAddress": "Habelschwerdter Allee 45, 14195 Berlin, Germany",
            "InstitutionalDepartmentName": (
                "Department of Education and Psychology, Neural Dynamics of "
                "Visual Cognition Lab (Cichy lab)"
            ),
            "Manufacturer": "Brain Products",
            "ManufacturersModelName": "BrainVision actiCHamp",
            "CapManufacturer": "EasyCap",
            "CapManufacturersModelName": "actiCAP 64Ch Standard-2",
            "PowerLineFrequency": 50,
            "EEGReference": "FCz (online); offline re-referenced as needed",
            "EEGGround": "AFz",
            "EEGPlacementScheme": "International 10-10",
            "RecordingType": "continuous",
            "SoftwareFilters": "n/a",
            "HardwareFilters": {
                "Highpass": {"CutoffFrequency": 0.01, "Unit": "Hz"},
                "Lowpass": {"CutoffFrequency": 100, "Unit": "Hz"},
            },
            "SoftwareVersions": "n/a",
            "DeviceSerialNumber": "n/a",
            "SubjectArtefactDescription": "n/a",
            "CogAtlasID": "n/a",
            "CogPOID": "n/a",
        }
    )
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
        f.write("\n")


def write_events_json(events_json_path: Path, task: str) -> None:
    """Write a per-run events.json sidecar."""
    levels = {
        "image": "Standard image presentation (training or test stimulus)",
        "target": "Random catch trial requiring a button press",
        "rest_marker": "Resting state start/end marker",
    }
    meta = {
        "onset": {"Description": "Event onset in seconds", "Units": "s"},
        "duration": {"Description": "Event duration in seconds", "Units": "s"},
        "sample": {"Description": "Sample index of the event onset (0-based)"},
        "value": {
            "Description": (
                "Trigger code from the BrainVision recording (1-99 cycle, "
                "does NOT uniquely identify the image)"
            )
        },
        "trial_type": {
            "Description": "Trial category",
            "Levels": levels,
        },
        "tot_img_number": {
            "Description": (
                "Global image ID from the THINGS-EEG2 stimulus set "
                "(1-16540 for train; 1-200 for test). 'n/a' on target trials."
            )
        },
        "img_category": {
            "Description": "Numeric category index from THINGS database"
        },
        "category_name": {
            "Description": (
                "THINGS category label, e.g. '01175_roller_coaster'"
            )
        },
        "block": {"Description": "Block index within this run"},
        "sequence": {"Description": "Sequence index within the block"},
        "img_in_sequence": {
            "Description": "Position of this image within its 20-image sequence"
        },
        "soa": {
            "Description": "Stimulus onset asynchrony as actually measured",
            "Units": "s",
        },
        # Verified from the authors' MATLAB data collection scripts
        # (00_data_collection/*/data_collection_*.m) which use PTB's
        # Screen() and KbName() APIs.
        "StimulusPresentation": {
            "OperatingSystem": "n/a",
            "SoftwareName": "Psychtoolbox-3",
            "SoftwareVersion": "n/a",
        },
    }
    with open(events_json_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def patch_coordsystem_jsons(eeg_dir: Path) -> None:
    """Add Fiducials* keys to coordsystem.json (mne-bids only writes
    AnatomicalLandmark*; BIDS validator wants both)."""
    for coord_path in Path(eeg_dir).glob("*_coordsystem.json"):
        with open(coord_path) as f:
            d = json.load(f)
        anat_coords = d.get("AnatomicalLandmarkCoordinates")
        anat_system = d.get("AnatomicalLandmarkCoordinateSystem")
        anat_units = d.get("AnatomicalLandmarkCoordinateUnits")
        eeg_desc = d.get("EEGCoordinateSystemDescription")
        if anat_coords and "FiducialsCoordinates" not in d:
            d["FiducialsCoordinates"] = anat_coords
        if anat_system and "FiducialsCoordinateSystem" not in d:
            d["FiducialsCoordinateSystem"] = anat_system
        if anat_units and "FiducialsCoordinateUnits" not in d:
            d["FiducialsCoordinateUnits"] = anat_units
        if eeg_desc and "FiducialsCoordinateSystemDescription" not in d:
            d["FiducialsCoordinateSystemDescription"] = (
                "Standard montage fiducials (NAS, LPA, RPA) from MNE's "
                "make_standard_montage. " + eeg_desc
            )
        if eeg_desc and "AnatomicalLandmarkCoordinateSystemDescription" not in d:
            d["AnatomicalLandmarkCoordinateSystemDescription"] = eeg_desc
        with open(coord_path, "w") as f:
            json.dump(d, f, indent=2)
            f.write("\n")


# --------------------------------------------------------------------------
# Per-run conversion
# --------------------------------------------------------------------------


def convert_run(
    src_eeg_dir: Path,
    src_beh_dir: Path,
    sub_id: str,
    ses_id: str,
    task: str,
    run: int | None,
    bids_root: Path,
    *,
    overwrite: bool,
    verbose: bool,
) -> bool:
    """Convert a single (subject, session, task, run) to BIDS-EEG.

    For task='train', `run` is 1..5 (mapped to source `task-train_part-XX`).
    For task='test', `run` is None (single run per session).
    For task='rest', `run` is 1 or 2 (mapped to source `task-rest1`/`task-rest2`).
    """
    sub_num = int(sub_id.replace("sub-", ""))
    ses_num = int(ses_id.replace("ses-", ""))
    sub_str = f"{sub_num:02d}"
    ses_str = f"{ses_num:02d}"

    if task == "train":
        if run is None:
            raise ValueError("train task requires a run number (1..5)")
        src_vhdr = (
            src_eeg_dir
            / f"sub-{sub_str}_ses-{ses_str}_task-train_part-{run:02d}_eeg.vhdr"
        )
        src_beh = (
            src_beh_dir
            / f"sub-{sub_str}_ses-{ses_str}_task-train_part-{run:02d}_beh.mat"
        )
    elif task == "test":
        src_vhdr = src_eeg_dir / f"sub-{sub_str}_ses-{ses_str}_task-test_eeg.vhdr"
        src_beh = src_beh_dir / f"sub-{sub_str}_ses-{ses_str}_task-test_beh.mat"
    elif task == "rest":
        if run not in (1, 2):
            raise ValueError(f"rest task requires run 1 or 2, got {run!r}")
        src_vhdr = (
            src_eeg_dir / f"sub-{sub_str}_ses-{ses_str}_task-rest{run}_eeg.vhdr"
        )
        src_beh = None
    else:
        raise ValueError(f"Unknown task: {task}")

    if not src_vhdr.exists():
        logger.warning("Missing source file: %s", src_vhdr)
        return False

    raw = load_brainvision(src_vhdr)
    raw.load_data(verbose=False)

    beh = None
    if src_beh and src_beh.exists():
        beh = parse_behavioral_mat(src_beh)

    events_df, annots = build_events_df(raw, beh, task)
    raw.set_annotations(annots, verbose=False)

    bids_path = mne_bids.BIDSPath(
        subject=sub_str,
        session=ses_str,
        task=task,
        run=f"{run:02d}" if run is not None else None,
        datatype="eeg",
        root=bids_root,
    )

    mne_bids.write_raw_bids(
        raw,
        bids_path,
        overwrite=overwrite,
        verbose=verbose,
        allow_preload=True,
        format="BDF",
    )

    # Overwrite events.tsv with our enriched version (mne_bids' version
    # only has onset/duration/trial_type/sample)
    events_tsv = bids_path.copy().update(suffix="events", extension=".tsv").fpath
    if events_tsv.exists() and len(events_df) > 0:
        col_order = ["onset", "duration", "sample", "value", "trial_type"]
        for c in events_df.columns:
            if c not in col_order:
                col_order.append(c)
        events_df[col_order].to_csv(
            events_tsv, sep="\t", index=False, na_rep="n/a"
        )

    # Enrich sidecars
    eeg_json = bids_path.copy().update(suffix="eeg", extension=".json").fpath
    enrich_eeg_sidecar(eeg_json, task, run)
    events_json = (
        bids_path.copy().update(suffix="events", extension=".json").fpath
    )
    write_events_json(events_json, task)

    return True


# --------------------------------------------------------------------------
# Sourcedata, derivatives, and stimuli
# --------------------------------------------------------------------------


def copy_sourcedata(input_dir: Path, bids_root: Path) -> None:
    """Copy the original BrainVision and behavioural files into /sourcedata/."""
    src_zips = sorted((input_dir / "source_zips").glob("sub-*.zip"))
    if not src_zips:
        logger.warning("No source zips found in %s/source_zips", input_dir)
        return
    sourcedata_dir = bids_root / "sourcedata"
    sourcedata_dir.mkdir(parents=True, exist_ok=True)
    for zf in src_zips:
        target = sourcedata_dir / zf.name
        if target.exists():
            continue
        shutil.copy2(zf, target)
        logger.info("  + sourcedata/%s", zf.name)


def copy_derivatives_preprocessed(input_dir: Path, bids_root: Path) -> None:
    """Copy the authors' preprocessed EEG zips into /derivatives/."""
    prep_zips = sorted((input_dir / "preprocessed_zips").glob("sub-*.zip"))
    if not prep_zips:
        logger.info("No preprocessed_zips folder")
        return
    deriv = bids_root / "derivatives" / "preprocessed_eeg"
    deriv.mkdir(parents=True, exist_ok=True)
    # Write a small dataset_description.json for the derivative
    desc = {
        "Name": "THINGS-EEG2 preprocessed EEG (Gifford et al. 2022)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "Gifford et al. 2022 preprocessing pipeline",
                "Description": (
                    "Author-released preprocessed train/test epochs as "
                    "distributed on OSF (https://osf.io/anp5v/). The 64-channel "
                    "version. Stored as one zip per subject."
                ),
                "CodeURL": "https://github.com/gifale95/eeg_encoding",
            }
        ],
        "SourceDatasets": [{"URL": "https://osf.io/anp5v/"}],
    }
    with open(deriv / "dataset_description.json", "w") as f:
        json.dump(desc, f, indent=2)
        f.write("\n")
    for zf in prep_zips:
        target = deriv / zf.name
        if target.exists():
            continue
        shutil.copy2(zf, target)
        logger.info("  + derivatives/preprocessed_eeg/%s", zf.name)


def copy_derivatives_resting_state(input_dir: Path, bids_root: Path) -> None:
    """Copy the authors' preprocessed resting state into /derivatives/."""
    rest_zips = sorted((input_dir / "resting_state_zips").glob("sub-*.zip"))
    if not rest_zips:
        logger.info("No resting_state_zips folder")
        return
    deriv = bids_root / "derivatives" / "resting_state"
    deriv.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": "THINGS-EEG2 preprocessed resting-state EEG (Gifford et al. 2022)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "Gifford et al. 2022 resting-state pipeline",
                "Description": (
                    "Author-released preprocessed resting-state EEG as "
                    "distributed on OSF (https://osf.io/6sg5e/)."
                ),
                "CodeURL": "https://github.com/gifale95/eeg_encoding",
            }
        ],
        "SourceDatasets": [{"URL": "https://osf.io/6sg5e/"}],
    }
    with open(deriv / "dataset_description.json", "w") as f:
        json.dump(desc, f, indent=2)
        f.write("\n")
    for zf in rest_zips:
        target = deriv / zf.name
        if target.exists():
            continue
        shutil.copy2(zf, target)
        logger.info("  + derivatives/resting_state/%s", zf.name)


def copy_stimuli(input_dir: Path, bids_root: Path) -> None:
    """Copy the image set into /stimuli/."""
    img_dir = input_dir / "image_set"
    if not img_dir.exists():
        logger.info("No image_set directory")
        return
    stim_dir = bids_root / "stimuli"
    stim_dir.mkdir(parents=True, exist_ok=True)
    for f in img_dir.iterdir():
        target = stim_dir / f.name
        if target.exists():
            continue
        shutil.copy2(f, target)
        logger.info("  + stimuli/%s", f.name)


# --------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------


def extract_subject_zip(
    zip_path: Path, dest_dir: Path
) -> Path:
    """Extract a subject zip into dest_dir and return the sub-XX root."""
    sub_name = zip_path.stem  # 'sub-01'
    target = dest_dir / sub_name
    if target.exists():
        return target
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Extracting %s...", zip_path.name)
    with zipfile.ZipFile(str(zip_path)) as zf:
        zf.extractall(str(dest_dir))
    return target


def _run_and_log(
    sub_name: str,
    ses_num: int,
    task: str,
    run: int | None,
    convert_fn,
) -> bool:
    """Invoke a conversion callable and log uniformly on failure."""
    ok = convert_fn()
    if not ok:
        run_label = f"run-{run:02d}" if run is not None else "(single)"
        logger.warning(
            "Failed: %s ses-%02d task-%s %s",
            sub_name,
            ses_num,
            task,
            run_label,
        )
    return ok


def convert_subject(
    sub_zip: Path,
    bids_root: Path,
    extract_dir: Path,
    *,
    overwrite: bool,
    verbose: bool,
    cleanup_extracted: bool = True,
) -> dict | None:
    """Convert a single subject (one zip) to BIDS, return participants row."""
    sub_root = extract_subject_zip(sub_zip, extract_dir)
    sub_name = sub_root.name  # 'sub-01'

    participant_age = None
    participant_sex = None
    n_sessions_done = 0
    fully_successful = False

    try:
        for ses_num in range(1, N_SESSIONS + 1):
            ses_dir = sub_root / f"ses-{ses_num:02d}"
            if not ses_dir.exists():
                continue
            eeg_dir = ses_dir / "eeg"
            beh_dir = ses_dir / "beh"
            if not eeg_dir.exists():
                continue

            # First behavioral .mat we find — read demographics
            if participant_age is None and beh_dir.exists():
                beh_files = sorted(beh_dir.glob("*_beh.mat"))
                if beh_files:
                    try:
                        beh = parse_behavioral_mat(beh_files[0])
                        participant_age = beh["subject_age"]
                        participant_sex = beh["subject_sex"]
                    except Exception as exc:
                        logger.debug(
                            "Could not parse beh demographics: %s", exc
                        )

            # 5 train runs
            for run in range(1, N_TRAIN_PARTS + 1):
                _run_and_log(
                    sub_name,
                    ses_num,
                    "train",
                    run,
                    lambda r=run: convert_run(
                        eeg_dir,
                        beh_dir,
                        sub_name,
                        f"ses-{ses_num:02d}",
                        "train",
                        r,
                        bids_root,
                        overwrite=overwrite,
                        verbose=verbose,
                    ),
                )

            # 1 test run (no run entity)
            _run_and_log(
                sub_name,
                ses_num,
                "test",
                None,
                lambda: convert_run(
                    eeg_dir,
                    beh_dir,
                    sub_name,
                    f"ses-{ses_num:02d}",
                    "test",
                    None,
                    bids_root,
                    overwrite=overwrite,
                    verbose=verbose,
                ),
            )

            # 2 rest runs
            for run in range(1, N_REST_RUNS + 1):
                _run_and_log(
                    sub_name,
                    ses_num,
                    "rest",
                    run,
                    lambda r=run: convert_run(
                        eeg_dir,
                        beh_dir,
                        sub_name,
                        f"ses-{ses_num:02d}",
                        "rest",
                        r,
                        bids_root,
                        overwrite=overwrite,
                        verbose=verbose,
                    ),
                )

            n_sessions_done += 1

            # Patch coordsystem.json files for this session
            bids_eeg_dir = bids_root / sub_name / f"ses-{ses_num:02d}" / "eeg"
            if bids_eeg_dir.exists():
                patch_coordsystem_jsons(bids_eeg_dir)

        fully_successful = n_sessions_done > 0
    finally:
        # Only clean up the extracted source folder if the whole subject
        # converted successfully; otherwise keep it for debugging.
        if cleanup_extracted and fully_successful and sub_root.exists():
            shutil.rmtree(sub_root, ignore_errors=True)

    sex_out = "n/a"
    if participant_sex:
        sex_out = str(participant_sex).strip().upper()[:1] or "n/a"
    return {
        "participant_id": sub_name,
        "age": participant_age if participant_age is not None else "n/a",
        "sex": sex_out,
        "n_sessions": n_sessions_done,
    }


def convert_thingseeg2(
    input_dir: Path,
    output_dir: Path,
    *,
    max_subjects: int | None = None,
    overwrite: bool = True,
    verbose: bool = False,
    skip_derivatives: bool = False,
    skip_stimuli: bool = False,
    skip_sourcedata: bool = False,
    keep_extracted: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    src_zips = sorted((input_dir / "source_zips").glob("sub-*.zip"))
    if not src_zips:
        raise FileNotFoundError(
            f"No source zips found at {input_dir / 'source_zips'}"
        )
    if len(src_zips) != EXPECTED_N_SUBJECTS and max_subjects is None:
        logger.warning(
            "Found %d source zips but THINGS-EEG2 publishes %d subjects",
            len(src_zips),
            EXPECTED_N_SUBJECTS,
        )
    if max_subjects:
        src_zips = src_zips[:max_subjects]
    logger.info("Found %d subject zips", len(src_zips))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_description(output_dir)
    write_readme(output_dir)
    write_participants_json(output_dir)

    extract_dir = input_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    participants: list[dict] = []
    for sub_zip in src_zips:
        logger.info("Processing %s", sub_zip.name)
        try:
            row = convert_subject(
                sub_zip,
                output_dir,
                extract_dir,
                overwrite=overwrite,
                verbose=verbose,
                cleanup_extracted=not keep_extracted,
            )
            if row is not None:
                participants.append(row)
        except Exception as exc:
            # Log traceback so failures are debuggable. Remove any partial
            # subject directory so a rerun starts clean.
            logger.error(
                "FAILED %s: %s", sub_zip.name, exc, exc_info=True
            )
            partial = output_dir / sub_zip.stem
            if partial.exists():
                logger.warning(
                    "Removing partial output at %s", partial
                )
                shutil.rmtree(partial, ignore_errors=True)
            continue

    if participants:
        pdf = pd.DataFrame(participants)
        pdf.to_csv(
            output_dir / "participants.tsv",
            sep="\t",
            index=False,
            na_rep="n/a",
        )
        logger.info("Wrote participants.tsv with %d rows", len(participants))

    # Sourcedata, derivatives and stimuli
    if not skip_sourcedata:
        logger.info("Copying sourcedata...")
        copy_sourcedata(input_dir, output_dir)
    if not skip_derivatives:
        logger.info("Copying derivatives...")
        copy_derivatives_preprocessed(input_dir, output_dir)
        copy_derivatives_resting_state(input_dir, output_dir)
    if not skip_stimuli:
        logger.info("Copying stimuli...")
        copy_stimuli(input_dir, output_dir)

    logger.info("Done. BIDS dataset at %s", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert THINGS-EEG2 to BIDS-EEG",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--max-subjects", "-n", type=int, default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--skip-derivatives", action="store_true")
    parser.add_argument("--skip-stimuli", action="store_true")
    parser.add_argument("--skip-sourcedata", action="store_true")
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Don't delete the extracted source folder after BIDS conversion",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        mne.set_log_level("WARNING")

    convert_thingseeg2(
        args.input,
        args.output,
        max_subjects=args.max_subjects,
        overwrite=not args.no_overwrite,
        verbose=args.verbose,
        skip_derivatives=args.skip_derivatives,
        skip_stimuli=args.skip_stimuli,
        skip_sourcedata=args.skip_sourcedata,
        keep_extracted=args.keep_extracted,
    )


if __name__ == "__main__":
    main()

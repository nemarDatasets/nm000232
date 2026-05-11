[![DOI](https://img.shields.io/badge/DOI-10.82901%2Fnemar.nm000232-blue)](https://doi.org/10.82901/nemar.nm000232)

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

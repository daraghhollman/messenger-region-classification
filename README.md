# MESSENGER Region Classification
Scripts involved in the creation, testing, and visualising a random forest as a
region classifier for the MESSENGER mission. Solar wind, magnetosheath, and
magnetosphere were differentiated with a ~98% testing accuracy. Steps to
reproduce this result are listed below.

The aim of this work was to produce a list of individual bow shock and
magnetopause crossings for the entire MESSENGER mission. The resulting list of
crossings, along with training data and other datasets are available via
[Zenodo](https://zenodo.org/records/15731194).

## Step-by-step guide to reproducing this work
This section outlines the order in which files were run to create this data
product. To execute these on your own machine, you will need to adjust any file
paths.

### Setup the hermpy package
This work relies heavily on the Python package
[hermpy](https://github.com/daraghhollman/hermpy) which requires some setup.
Follow the setup guide for hermpy found in the Github README.

This includes:
- Downloading MESSENGER MAG data
- Downloading the Philpott et al. (2020) crossing interval list
- Setting up file paths

### Create the training dataset
The crossing intervals list is used as a reference with which to construct the
training dataset. The script `sampling/get_samples_multiprocessed.py` is used
to take samples of solar wind, magnetosheath, and magnetosphere. This script
creates 4 files:
- solar_wind_samples.csv
- bs_magnetosheath_samples.csv (magnetosheath samples extracted adjacent to bow shock intervals)
- mp_magnetosheath_samples.csv (magnetosheath samples extracted adjacent to magnetopause intervals)
- magnetosphere_samples.csv

These files are loaded by the script `modelling/train_model.py` when training
the model, and features are extracted from each sample.

### Model Training
The model is created using the script `modelling/train_model.py`. A copy of the
combined training dataset is saved as `zenodo-data/training_data.csv`
(zenodo-data is not included in this repository as the files inside are too
large. However, this is available at the link in the first section of this
README). The model is fit multiple times (with shuffled training / testing
split and seed) to determine variance in the method. The default number is 10,
though this is adjustable. All models are saved for the purposes of visualising
feature importance and confusion matrices, though only one model is chosen to
be applied to the data as a whole.

### Model Application
The model is then applied to the full mission with
`application/get_probabilities.py`. Time stamps and probabilities for each of
the three classes are saved to one file `data/model_raw_output.csv`.

Direct testing of the model on individual cases is done using
`applicastion/apply_model.py`.

### Determining Crossings and Post-processing
Crossings are then determined from the model output using
`application/find_crossings_from_probabilities.py`. This places crossings where
changes in region occur, and saves them to `data/new_crossings.csv`. The
regions (now collated into contiguous blocks)

Post processing of this crossing list is done in
`post-processing/1_ensure_start_and_end.py` and
`post-processing/2_include_hiddent_crossings.py` in that order specifically (as
the second relies on the output of the first). The finalised list of crossings
is output to `zenodo-data/hollman_2025_post_processing_list.csv`.

## References

Philpott, L. C., Johnson, C. L., Anderson, B. J., & Winslow, R. M. (2020). The
shape of mercury’s magnetopause: The picture from messenger magnetometer
observations and future prospects for BepiColombo. Journal of Geophysical
Research: Space Physics, 125 (5), e2019JA027544. (e2019JA027544541
10.1029/2019JA027544) doi: https://doi.org/10.1029/2019JA027544

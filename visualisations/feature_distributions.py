"""
A script to investigate the distribution of features in the training data (the assumed ground truth) as compared
to how we would now define the regions with these new crossings.

This would be computationally intensive to do for all crossings. We can instead get the feature distributions near N
random crossing intervals.
"""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from hermpy import boundaries, mag, utils
from hermpy.plotting import wong_colours

# Load Philpott intervals
print("Loading Philpott intervals")
philpott_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=True
)

# Load the MESSENGER mission
print("Loading MESSENGER data")
messenger_mag = mag.Load_Mission("/home/daraghhollman/Main/data/mercury/messenger_mag")

# Fetch training data
print("Searching for training data")
# Limit to n random intervals
n_intervals = 200
interval_buffer = dt.timedelta(minutes=10)
random_intervals = philpott_intervals.sample(frac=1).iloc[0:n_intervals]

random_bow_shock_in_intervals = random_intervals.loc[
    random_intervals["Type"] == "BS_IN"
]
random_bow_shock_out_intervals = random_intervals.loc[
    random_intervals["Type"] == "BS_OUT"
]
random_magnetopause_in_intervals = random_intervals.loc[
    random_intervals["Type"] == "MP_IN"
]
random_magnetopause_out_intervals = random_intervals.loc[
    random_intervals["Type"] == "MP_OUT"
]


def get_training_data(intervals):

    region_before_times = []
    region_after_times = []

    for i, interval in intervals.iterrows():
        previous_interval = philpott_intervals.loc[i - 1]
        next_interval = philpott_intervals.loc[i + 1]

        # Region before the interval!
        data_start = interval["Start Time"] - interval_buffer
        data_end = interval["Start Time"]

        # Ensure region doesn't go into another interval
        if data_start < previous_interval["End Time"]:
            data_start = previous_interval["End Time"]

        # Save start and end times
        region_before_times.append((data_start, data_end))

        # Region after the interval!
        data_start = interval["End Time"]
        data_end = interval["End Time"] + interval_buffer

        # Ensure region doesn't go into another interval
        if data_end > next_interval["Start Time"]:
            data_end = next_interval["Start Time"]

        # Save start and end times
        region_after_times.append((data_start, data_end))

    # Get the data for all times within each region
    region_before_data = pd.concat(
        [
            messenger_mag.loc[messenger_mag["date"].between(start, end)]
            for start, end in region_before_times
        ]
    )
    region_after_data = pd.concat(
        [
            messenger_mag.loc[messenger_mag["date"].between(start, end)]
            for start, end in region_after_times
        ]
    )

    return region_before_data, region_after_data


solar_wind_original_data_1, magnetosheath_original_data_1 = get_training_data(
    random_bow_shock_in_intervals
)
magnetosheath_original_data_2, solar_wind_original_data_2 = get_training_data(
    random_bow_shock_out_intervals
)

magnetosphere_original_data_1, magnetosheath_original_data_3 = get_training_data(
    random_magnetopause_out_intervals
)
magnetosheath_original_data_4, magnetosphere_original_data_2 = get_training_data(
    random_magnetopause_in_intervals
)

solar_wind_original_data = pd.concat(
    [solar_wind_original_data_1, solar_wind_original_data_2]
)
magnetosheath_original_data = pd.concat(
    [
        magnetosheath_original_data_1,
        magnetosheath_original_data_2,
        magnetosheath_original_data_3,
        magnetosheath_original_data_4,
    ]
)
magnetosphere_original_data = pd.concat(
    [magnetosphere_original_data_1, magnetosheath_original_data_2]
)


# We can now look for our new data
print("Searching through new regions")
# We saved the regions determined when finding the crossings
# These are easier to work with in this instance than the crossings themselves
hollman_regions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_regions.csv"
)
# Again we limit to some n random regions
n_regions = 5000
random_regions = hollman_regions.sample(frac=1).iloc[0:n_regions]


def get_new_region_data(regions, label):

    labelled_regions = regions.loc[regions["Label"] == label]

    region_times = []
    for i, region in labelled_regions.iterrows():

        region_times.append((region["Start Time"], region["End Time"]))

    region_data = pd.concat(
        [
            messenger_mag.loc[messenger_mag["date"].between(start, end)]
            for start, end in region_times
        ]
    )

    return region_data


solar_wind_new_data = get_new_region_data(random_regions, "Solar Wind")
magnetosheath_new_data = get_new_region_data(random_regions, "Magnetosheath")
magnetosphere_new_data = get_new_region_data(random_regions, "Magnetosphere")


# Make a grid of features we wish to look at
fig, axes = plt.subplots(3, 4, sharex=True, sharey="row")

original_data = [
    solar_wind_original_data,
    magnetosheath_original_data,
    magnetosphere_original_data,
]
new_data = [solar_wind_new_data, magnetosheath_new_data, magnetosphere_new_data]

column_variables = ["|B|", "Bx", "By", "Bz"]

bins = np.linspace(-150, 150, 30)

for row_index in range(len(axes[:, 0])):
    for column_index in range(len(axes[0])):

        ax = axes[row_index][column_index]

        ax.hist(
            original_data[row_index][column_variables[column_index]],
            density=True,
            histtype="step",
            color=wong_colours["black"],
            linewidth=2,
            alpha=0.7,
            bins=bins,
            label=f"Training Data",
        )
        ax.hist(
            new_data[row_index][column_variables[column_index]],
            density=True,
            histtype="step",
            color=wong_colours["light blue"],
            linewidth=2,
            alpha=0.7,
            bins=bins,
            label=f"New Regions Data",
        )

        ax.set_title(
            f"N(Training Data){len(original_data[row_index][column_variables[column_index]])}\nN(New Regions Data){len(new_data[row_index][column_variables[column_index]])}"
        )

axes[0][0].legend()

axes[0][0].set_ylabel("Solar Wind\nData Density")
axes[1][0].set_ylabel("Magnetosheath\nData Density")
axes[2][0].set_ylabel("Magnetosphere\nData Density")

axes[-1][0].set_xlabel("|B| [nT]")
axes[-1][1].set_xlabel("Bx [nT]")
axes[-1][2].set_xlabel("By [nT]")
axes[-1][3].set_xlabel("Bz [nT]")

plt.show()

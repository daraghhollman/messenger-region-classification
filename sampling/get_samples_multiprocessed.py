"""
In this script, we search through each boundary in the Philpott boundary interval list and extract many samples
of time series data before and after it.
To save on compute time, we take X random samples of size Y, within time Z of boundary edges.
"""

import csv
import datetime as dt
import multiprocessing
import os
import random
import shutil
import tempfile

import numpy as np
import pandas as pd
import scipy.stats
import spiceypy as spice
from hermpy import boundaries, mag, trajectory, utils
from tqdm import tqdm

# Load previous progress from csv
sw_output = "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/solar_wind_samples.csv"
bs_magnetosheath_output = "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/bs_magnetosheath_samples.csv"
mp_magnetosheath_output = "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/mp_magnetosheath_samples.csv"
magnetosphere_output = "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/magnetosphere_samples.csv"

output_files = [
    sw_output,
    bs_magnetosheath_output,
    mp_magnetosheath_output,
    magnetosphere_output,
]

for file in output_files:
    if not os.path.exists(file):
        os.mknod(file)

try:
    previous_indices = [
        pd.read_csv(file).iloc[-1]["Boundary ID"] for file in output_files
    ]

    last_index = np.max(previous_indices) + 1
except pd.errors.EmptyDataError:
    last_index = -1


def Get_Random_Sample(
    data,
    search_start,
    search_end,
    length,
    boundary_type,
    sample_location,
):

    # Choose random start point between
    sample_start = search_start + dt.timedelta(
        seconds=random.randint(0, int((search_end - search_start).total_seconds()))
    )
    sample_end = sample_start + length

    # Sample data
    sample_data = data.loc[data["date"].between(sample_start, sample_end)]

    # Label the sample based on the type of boundary crossing interval
    labels = {
        ("BS_IN", "before"): "Solar Wind",
        ("BS_IN", "after"): "BS Magnetosheath",
        ("BS_OUT", "before"): "BS Magnetosheath",
        ("BS_OUT", "after"): "Solar Wind",
        ("MP_IN", "before"): "MP Magnetosheath",
        ("MP_IN", "after"): "Magnetosphere",
        ("MP_OUT", "before"): "Magnetosphere",
        ("MP_OUT", "after"): "MP Magnetosheath",
    }

    sample_label = labels.get((boundary_type, sample_location), "Unknown")

    if len(sample_data) == 0:
        sample_features = {"Label": sample_label}
        return sample_features

    # Get features
    sample_features = Get_Sample_Features(sample_data)
    sample_features["Label"] = sample_label

    return sample_features


def Get_Sample_Features(data):

    mean = np.mean([data["|B|"], data["Bx"], data["By"], data["Bz"]], axis=1)
    median = np.median([data["|B|"], data["Bx"], data["By"], data["Bz"]], axis=1)
    std = np.std([data["|B|"], data["Bx"], data["By"], data["Bz"]], axis=1)
    skew = scipy.stats.skew([data["|B|"], data["Bx"], data["By"], data["Bz"]], axis=1)
    kurtosis = scipy.stats.kurtosis(
        [data["|B|"], data["Bx"], data["By"], data["Bz"]], axis=1
    )

    data_middle = data.iloc[round(len(data) / 2)]

    sample_middle_position = np.array(
        [
            data_middle["X MSM' (radii)"],
            data_middle["Y MSM' (radii)"],
            data_middle["Z MSM' (radii)"],
        ]
    )

    local_time = trajectory.Local_Time(sample_middle_position)
    latitude = trajectory.Latitude(sample_middle_position)
    magnetic_latitude = trajectory.Magnetic_Latitude(sample_middle_position)
    sample_middle_position = np.array(
        [
            data_middle["X MSM' (radii)"],
            data_middle["Y MSM' (radii)"],
            data_middle["Z MSM' (radii)"],
        ]
    )

    local_time = trajectory.Local_Time(sample_middle_position)
    latitude = trajectory.Latitude(sample_middle_position)
    magnetic_latitude = trajectory.Magnetic_Latitude(sample_middle_position)

    with spice.KernelPool(utils.User.METAKERNEL):
        et = spice.str2et(data_middle["date"].strftime("%Y-%m-%d %H:%M:%S"))
        mercury_position, _ = spice.spkpos("MERCURY", et, "J2000", "NONE", "SUN")

        # The way the code is written in hermpy, it would be much faster to do
        # this heliocentric distance calculation for all samples at the end,
        # rather than during sample collection.
        heliocentric_distance = np.sqrt(
            mercury_position[0] ** 2
            + mercury_position[1] ** 2
            + mercury_position[2] ** 2
        )
        heliocentric_distance = utils.Constants.KM_TO_AU(heliocentric_distance)

    return {
        # Time identifiers
        "Sample Start": data["date"].iloc[0],
        "Sample End": data["date"].iloc[-1],
        # Parameters
        "Mean": mean,
        "Median": median,
        "Standard Deviation": std,
        "Skew": skew,
        "Kurtosis": kurtosis,
        "Heliocentric Distance (AU)": heliocentric_distance,
        "Local Time (hrs)": local_time,
        "Latitude (deg.)": latitude,
        "Magnetic Latitude (deg.)": magnetic_latitude,
        "X MSM' (radii)": data_middle["X MSM' (radii)"],
        "Y MSM' (radii)": data_middle["Y MSM' (radii)"],
        "Z MSM' (radii)": data_middle["Z MSM' (radii)"],
    }


def Process_Crossing(inputs):
    i, crossing_interval = inputs

    if crossing_interval["Type"] == "DATA_GAP":
        return None

    # We define the eariest possible sample start
    # and latest possible sample end.
    # Making sure to never go past the next boundary.
    earliest_sample_start_before = crossing_interval["Start Time"] - search_distance
    latest_sample_start_before = crossing_interval["Start Time"] - sample_length

    if i > 0:
        if earliest_sample_start_before < crossing_intervals.loc[i - 1]["End Time"]:
            earliest_sample_start_before = crossing_intervals.loc[i - 1]["End Time"]

    earliest_sample_start_after = crossing_interval["End Time"]
    latest_sample_start_after = (
        crossing_interval["End Time"] + search_distance - sample_length
    )

    if i > 0:
        if latest_sample_start_after > crossing_intervals.loc[i + 1]["Start Time"]:
            latest_sample_start_after = (
                crossing_intervals.loc[i + 1]["Start Time"] - sample_length
            )

    if earliest_sample_start_before > latest_sample_start_before:
        raise ValueError(
            f"Sample start ({earliest_sample_start_before}) is after sample end ({latest_sample_start_before})!"
        )

    if earliest_sample_start_after > latest_sample_start_after:
        raise ValueError(
            f"Sample start ({earliest_sample_start_after}) is after sample end ({latest_sample_start_after})!"
        )

    # Load data
    surrounding_data = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG_FULL"],
        earliest_sample_start_before,
        latest_sample_start_after,
        average=None,
    )

    samples = []
    number_of_samples = 10
    for _ in range(number_of_samples):
        # Save sample before BCI
        sample_before = Get_Random_Sample(
            surrounding_data,
            earliest_sample_start_before,
            latest_sample_start_before,
            sample_length,
            str(crossing_interval["Type"]),
            "before",
        )

        # Save sample after
        sample_after = Get_Random_Sample(
            surrounding_data,
            earliest_sample_start_after,
            latest_sample_start_after,
            sample_length,
            str(crossing_interval["Type"]),
            "after",
        )

        for sample in [sample_before, sample_after]:
            sample["Boundary ID"] = i

        samples.append([sample_before, sample_after])

    return samples


##########################
##### MAIN SCRIPT ########
##########################
sample_length = dt.timedelta(seconds=10)  # How long is each sample
search_distance = dt.timedelta(
    minutes=10
)  # How far from the boundary can we collect samples

# Load boundary crossings
crossing_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=True
)

# Loop through boundary crossing intervals
# Randomly take a sample of data within some distance from the boundary.

process_items = [
    (i, crossing_interval) for i, crossing_interval in crossing_intervals.iterrows()
]

print(f"Continuing from crossing id: {last_index + 1}")

process_items = process_items[last_index + 1 :]


# Some fancy code to enable us to start and stop the script if needed, without losing progress
def Safely_Append_Row(output_file, sample):

    # Write to a temporary file first.
    # That way, if there are any corruptions, they won't occur in the main file
    tmp_file_name = ""
    with tempfile.NamedTemporaryFile("w", delete=False, newline="") as tmp_file:
        writer = csv.writer(tmp_file)
        writer.writerow(sample)
        tmp_file_name = tmp_file.name

    # Append the temp fiile contents atomically
    # i.e. the write happens at once, and errors can't occur from partial writes
    with (
        open(output_file, "a", newline="") as out_file,
        open(tmp_file_name, "r") as tmp_file,
    ):
        shutil.copyfileobj(tmp_file, out_file)

        # Flush python's buffer and force os to flush file to disk
        out_file.flush()
        os.fsync(out_file.fileno())

    # Clean tmp file
    os.remove(tmp_file.name)


with multiprocessing.Pool(
    int(input(f"Number of cores? __ / {multiprocessing.cpu_count()} "))
) as pool:

    for samples_taken in tqdm(
        pool.imap(Process_Crossing, process_items),
        desc="Processing crossings",
        total=len(process_items),
    ):
        if not isinstance(samples_taken, list):
            continue

        # Save samples
        for row in samples_taken:
            for sample in [row[0], row[1]]:

                match sample["Label"]:
                    case "Solar Wind":
                        output_file = sw_output
                    case "BS Magnetosheath":
                        output_file = bs_magnetosheath_output
                    case "MP Magnetosheath":
                        output_file = mp_magnetosheath_output
                    case "Magnetosphere":
                        output_file = magnetosphere_output

                    case _:
                        raise ValueError(f"Unknown sample label: {sample['Label']}")

                # If there is no data, only two columns:
                # boundary id, and label
                if len(sample) == 2:
                    """
                    with open(output_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([np.nan] * 14 + list(sample.values()))
                    """
                    Safely_Append_Row(
                        output_file, [np.nan] * 14 + list(sample.values())
                    )
                    continue

                # If the file doesn't exist, create it
                if not os.path.exists(output_file):
                    os.mknod(output_file)

                    """
                    with open(output_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(list(sample.keys()))
                    """
                    Safely_Append_Row(output_file, list(sample.keys()))

                else:
                    try:
                        pd.read_csv(output_file)
                    except pd.errors.EmptyDataError:
                        """
                        with open(output_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(list(sample.keys()))
                        """
                        Safely_Append_Row(output_file, list(sample.keys()))

                    """
                    with open(output_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(sample.values())
                    """
                    Safely_Append_Row(output_file, sample.values())

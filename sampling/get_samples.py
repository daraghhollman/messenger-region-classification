"""
In this script, we search through each boundary in the Philpott boundary interval list and extract many samples
of time series data before and after it.
To save on compute time, we could take X random samples of size Y, within time Z of boundary edges.
"""

import csv
import datetime as dt
import os
import random

import numpy as np
import pandas as pd
import scipy.stats
import spiceypy as spice
from hermpy import boundaries, mag, trajectory, utils


def main():
    sample_length = dt.timedelta(seconds=10)  # How long is each sample
    search_distance = dt.timedelta(
        minutes=10
    )  # How far from the boundary can we collect samples

    # Load boundary crossings
    crossing_intervals = boundaries.Load_Crossings(
        utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False
    )

    # Loop through boundary crossing intervals
    # Randomly take a sample of data within some distance from the boundary.

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

    try:
        previous_indices = [
            pd.read_csv(file).iloc[-1]["Boundary ID"] for file in output_files
        ]

        last_index = np.max(previous_indices) + 1
    except IndexError:
        last_index = -1

    for i, crossing_interval in crossing_intervals.iterrows():

        if i <= last_index:
            continue

        print(f"Processing crossing {i}")

        # We define the eariest possible sample start
        # and latest possible sample end.
        # Making sure to never go past the next boundary.
        earliest_sample_start_before = crossing_interval["Start Time"] - search_distance
        latest_sample_start_before = crossing_interval["Start Time"] - sample_length

        earliest_sample_start_after = crossing_interval["End Time"]
        latest_sample_start_after = (
            crossing_interval["End Time"] + search_distance - sample_length
        )

        # Load data
        surrounding_data = mag.Load_Between_Dates(
            utils.User.DATA_DIRECTORIES["MAG_FULL"],
            earliest_sample_start_before,
            latest_sample_start_after,
            average=None,
        )

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

        # Save samples
        for sample in [sample_before, sample_after]:

            sample["Boundary ID"] = i

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

            # If the file doesn't exist, create it
            if not os.path.exists(output_file):
                os.mknod(output_file)

                with open(output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(list(sample.keys()))

            else:
                with open(output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(sample.values())


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


if __name__ == "__main__":
    main()

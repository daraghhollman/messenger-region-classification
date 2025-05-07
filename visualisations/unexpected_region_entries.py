"""
A script to investigate crossing groups which have unknown regions at their
begining or end, groups which have the opposite region at the begining or end
based on their interval, and finally, intervals which contain groups of the 3rd
'incorrect' class.
"""

import numpy as np
import pandas as pd
from hermpy import boundaries, utils

# Load the regions
new_regions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_regions.csv"
)

# Load the Philpott crossings
philpott_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False
)

# The first column of our data marks the index of that region
# within the group. Therefore we can regain our groups by
# splitting the regions when this column is 0
# Split indices marks the start of each new region
print("Spliting region data back into groups")
split_indices = np.where(new_regions.iloc[:, 0] == 0)[0]

crossing_groups = [
    new_regions[start:end] for start, end in zip(split_indices[:-1], split_indices[1:])
]
print(f"{len(crossing_groups)} groups")

"""

# For each group, we want to find which region is at the start, and which
# region is at the end. If this is an unknown region, we increment the unknown
# counter. If not, we compare to the interval in that group (next interval for
# the first region, previous interval for the last)

incorrect_starting_regions = 0
incorrect_ending_regions = 0
start_and_end_incorrect = 0

expected_interval_starts = {
    "BS_IN": "Solar Wind",
    "BS_OUT": "Magnetosheath",
    "MP_IN": "Magnetosheath",
    "MP_OUT": "Magnetosphere",
}
expected_interval_ends = {
    "BS_IN": "Magnetosheath",
    "BS_OUT": "Solar Wind",
    "MP_IN": "Magnetosphere",
    "MP_OUT": "Magnetosheath",
}

for group in crossing_groups:

    first_region = group.iloc[0]
    last_region = group.iloc[-1]

    # By definition, the first and last region must be outide all Philpott
    # intervals. Hence, we need only compare the first region with the next
    # Philpott interval, and the last region with only the previous.
    interval_after_first_region = philpott_intervals.loc[
        philpott_intervals["Start Time"] > first_region["Start Time"]
    ].iloc[0]

    interval_before_last_region = philpott_intervals.loc[
        philpott_intervals["End Time"] < last_region["End Time"]
    ].iloc[-1]

    a = b = False

    if (
        first_region["Label"]
        != expected_interval_starts[interval_after_first_region["Type"]]
    ):
        a = True
        incorrect_starting_regions += 1

    if (
        last_region["Label"]
        != expected_interval_ends[interval_before_last_region["Type"]]
    ):
        b = True
        incorrect_ending_regions += 1

    if a & b:
        start_and_end_incorrect += 1

print(f"Groups with both start and end incorrect: {start_and_end_incorrect}")
print(
    f"Groups with only start incorrect: {incorrect_starting_regions - start_and_end_incorrect}"
)
print(
    f"Groups with only end incorrect: {incorrect_ending_regions - start_and_end_incorrect}"
)

"""

# We also want to loop through the Philpott intervals and check if any incorrect regions appear within the interval.
new_regions["Start Time"] = pd.to_datetime(new_regions["Start Time"])
new_regions["End Time"] = pd.to_datetime(new_regions["End Time"])

unexpected_instances = 0
unexpected_indices = []
unexpected_time = 0

for index, interval in philpott_intervals.iterrows():

    regions_within_interval = new_regions.loc[
        (
            new_regions["Start Time"].between(
                interval["Start Time"], interval["End Time"]
            )
        )
        | (
            new_regions["End Time"].between(
                interval["Start Time"], interval["End Time"]
            )
        )
    ]

    unexpected_regions = pd.DataFrame()

    match interval["Type"]:

        case "BS_IN" | "BS_OUT":
            unexpected_regions = regions_within_interval.loc[
                regions_within_interval["Label"] == "Magnetosphere"
            ]

        case "MP_IN" | "MP_OUT":
            unexpected_regions = regions_within_interval.loc[
                regions_within_interval["Label"] == "Solar Wind"
            ]

    if len(unexpected_regions) != 0:
        # Count the number of seconds
        time = unexpected_regions["Duration (seconds)"].sum()
        # Count the number of instances
        unexpected_instances += 1
        unexpected_time += time

        unexpected_indices.append(index)

print(unexpected_instances)
print(unexpected_indices)
print(f"{unexpected_time} seconds")

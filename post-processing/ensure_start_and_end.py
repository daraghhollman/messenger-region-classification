"""
Takes raw outputs from the model and compares each application starting and
ending region to what is expected by Philpott.
"""

import numpy as np
import pandas as pd
from hermpy import boundaries, utils

# The region classifications are the raw output
regions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_regions.csv"
)

# We need to split the list of regions back up into a list of applications. The
# first column of our list marks the index of the region within the group.
# Therefore we can regain our groups by splitting the regions where this column
# == 0.
# Each split index marks the start of each new region.
split_indices = np.where(regions.iloc[:, 0] == 0)[0]
split_indices = np.append(split_indices, len(regions))  # Include last group

region_groups = [
    regions[start:end] for start, end in zip(split_indices[:-1], split_indices[1:])
]

# We want to compare with the Philpott intervals
philpott_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False
)

new_region_groups = []

# For each group, we want to find the associated Philpott intervals.
for group_regions in region_groups:

    first_region = group_regions.iloc[0]
    last_region = group_regions.iloc[-1]

    application_start = first_region["Start Time"]
    application_end = last_region["End Time"]

    intervals_within_application = philpott_intervals.loc[
        philpott_intervals["Start Time"].between(application_start, application_end)
    ]

    transition_map = {
        "BS_IN": ("Solar Wind", "Magnetosheath"),
        "BS_OUT": ("Magnetosheath", "Solar Wind"),
        "MP_IN": ("Magnetosheath", "Magnetosphere"),
        "MP_OUT": ("Magnetosphere", "Magnetosheath"),
    }

    if len(intervals_within_application) == 1:

        interval_type = intervals_within_application.iloc[0]["Type"]

        expected_starting_region, expected_ending_region = transition_map.get(
            interval_type, ("Error", "Error")
        )

    else:
        start_type = intervals_within_application.iloc[0]["Type"]
        end_type = intervals_within_application.iloc[-1]["Type"]

        expected_starting_region = transition_map.get(start_type, ("Error", None))[0]
        expected_ending_region = transition_map.get(end_type, (None, "Error"))[1]

    if first_region["Label"] != expected_starting_region:
        group_regions.loc[group_regions.index[0], "Label"] = expected_starting_region

    if last_region["Label"] != expected_ending_region:
        group_regions.loc[group_regions.index[-1], "Label"] = expected_ending_region

    new_region_groups.append(group_regions)

post_processed_regions = pd.concat(new_region_groups)

post_processed_regions.to_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/post-processing/bookend_regions_processed.csv"
)

"""
Takes raw outputs from the model and compares each application starting and
ending region to what is expected by Philpott.
"""

import datetime as dt

import numpy as np
import pandas as pd
from hermpy import boundaries, utils

# The region classifications are the raw output
regions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_regions.csv",
    parse_dates=["Start Time", "End Time"],
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
number_of_regions_changed = 0

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

        # If the first or last region span across the crossing interval, we instead
        # want to split it at the outermost part of the interval

        if (
            first_region["End Time"]
            > intervals_within_application.iloc[0]["Start Time"]
        ):

            new_region = pd.DataFrame(
                [
                    {
                        "Start Time": intervals_within_application.iloc[0][
                            "Start Time"
                        ],
                        "End Time": first_region["End Time"],
                        "Label": first_region["Label"],
                    }
                ]
            )

            # Modify the first region's end time
            group_regions.iloc[0, group_regions.columns.get_loc("End Time")] = (
                intervals_within_application.iloc[0]["Start Time"]
            )

            # Insert new region just after first
            group_regions = pd.concat(
                [group_regions.iloc[:1], new_region, group_regions.iloc[1:]],
                ignore_index=True,
            )

        if last_region["Start Time"] < intervals_within_application.iloc[0]["End Time"]:

            new_region = pd.DataFrame(
                [
                    {
                        "Start Time": last_region["Start Time"],
                        "End Time": intervals_within_application.iloc[0]["End Time"],
                        "Label": last_region["Label"],
                    }
                ]
            )

            # Modify the last region's start time
            group_regions.iloc[-1, group_regions.columns.get_loc("Start Time")] = (
                intervals_within_application.iloc[0]["End Time"]
            )

            # Insert new region just before last
            group_regions = pd.concat(
                [group_regions.iloc[:-1], new_region, group_regions.iloc[-1:]],
                ignore_index=True,
            )

    else:

        start_type = intervals_within_application.iloc[0]["Type"]
        end_type = intervals_within_application.iloc[-1]["Type"]

        expected_starting_region = transition_map.get(start_type, ("Error", None))[0]
        expected_ending_region = transition_map.get(end_type, (None, "Error"))[1]

        # If the first or last region span across the crossing interval, we instead
        # want to split it at the outermost part of the interval

        if (
            first_region["End Time"]
            > intervals_within_application.iloc[0]["Start Time"]
        ):

            new_region = pd.DataFrame(
                [
                    {
                        "Start Time": intervals_within_application.iloc[0][
                            "Start Time"
                        ],
                        "End Time": first_region["End Time"],
                        "Label": first_region["Label"],
                    }
                ]
            )

            # Modify the first region's end time
            group_regions.iloc[0, group_regions.columns.get_loc("End Time")] = (
                intervals_within_application.iloc[0]["Start Time"]
            )

            # Insert new region just after first
            group_regions = pd.concat(
                [group_regions.iloc[:1], new_region, group_regions.iloc[1:]],
                ignore_index=True,
            )

        if (
            last_region["Start Time"]
            < intervals_within_application.iloc[-1]["End Time"]
        ):

            new_region = pd.DataFrame(
                [
                    {
                        "Start Time": last_region["Start Time"],
                        "End Time": intervals_within_application.iloc[-1]["End Time"],
                        "Label": last_region["Label"],
                    }
                ]
            )

            # Modify the last region's start time
            group_regions.iloc[-1, group_regions.columns.get_loc("Start Time")] = (
                intervals_within_application.iloc[-1]["End Time"]
            )

            # Insert new region just before last
            group_regions = pd.concat(
                [group_regions.iloc[:-1], new_region, group_regions.iloc[-1:]],
                ignore_index=True,
            )

    # After insertion, we need to redefine regions
    first_region_index = group_regions.index[0]
    last_region_index = group_regions.index[-1]

    application_counted = False
    if group_regions.at[first_region_index, "Label"] != expected_starting_region:
        group_regions.at[first_region_index, "Label"] = expected_starting_region
        number_of_regions_changed += 1
        application_counted = True

    if group_regions.at[last_region_index, "Label"] != expected_ending_region:
        group_regions.at[last_region_index, "Label"] = expected_ending_region
        if not application_counted:
            number_of_regions_changed += 1

    # We want to combine neighbouring regions of the same type
    # This dramatically slows down the code. Is there a faster way?
    combined_regions = [group_regions.iloc[0].copy()]
    for i in range(1, len(group_regions)):
        current = group_regions.iloc[i]
        previous = combined_regions[-1]

        if current["Label"] == previous["Label"]:
            # Extend the previous region's end time
            combined_regions[-1]["End Time"] = current["End Time"]
        else:
            combined_regions.append(current.copy())

    group_regions = pd.DataFrame(combined_regions)

    new_region_groups.append(group_regions)

post_processed_regions = pd.concat(new_region_groups)

post_processed_regions.to_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/post-processing/1_bookend_regions_processed.csv"
)

print(f"{number_of_regions_changed} regions changed")
print(f"{number_of_regions_changed / len(new_region_groups)}%")

"""
Our models output a csv file with times and probabilities. This script reads that and finds the crossings
"""

import collections
import datetime as dt

import numpy as np
import pandas as pd
from tqdm import tqdm

# Load the data
print("Loading Model Output")
model_output = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/model_raw_output.csv"
)
print("Output Loaded")

# Convert from string to datetime
model_output["Time"] = pd.to_datetime(model_output["Time"], format="ISO8601")

# Search for the gaps longer than 1 second and split
times = model_output["Time"].to_numpy()
probabilities = {
    "Solar Wind": model_output["P(SW)"].to_numpy(),
    "Magnetosheath": model_output["P(MSh)"].to_numpy(),
    "Magnetosphere": model_output["P(MSp)"].to_numpy(),
}

time_differences = np.diff(times).astype("timedelta64[s]").astype(int)

# Jump indices mark the start of the next crossing group
# (This is why we add 1)
jump_indices = np.where(time_differences > 1)[0] + 1
negative_jump_indices = np.where(time_differences < -1)[0] + 1
no_jump = np.where(abs(time_differences) < 1)[0] + 1

if len(no_jump) != 0:
    raise ValueError(
        "Duplicate Predictions in model output!\n" + f"Data at indices: {no_jump}"
    )

# If there are negative indices, this is alright, but means we have overlap
# in our groups. We can simply merge these groups by removing duplicate rows.

# There is exactly 1 second between each consecutive measurement.
# At each negative jump index, we can hence find the time difference
# in seconds, and remove that many rows from the dataframe
# We must do this in reverse as the indices will change each time
for i in reversed(negative_jump_indices):
    time_jump = (times[i - 1] - times[i]).astype("timedelta64[s]").astype(int)
    indices_to_delete = np.arange(i, i + time_jump + 1)
    indices_to_delete = indices_to_delete[indices_to_delete < len(times)]

    times = np.delete(times, indices_to_delete)
    for key in probabilities:
        probabilities[key] = np.delete(probabilities[key], indices_to_delete)

# Recalculate jump_indices now that times has changed
time_differences = np.diff(times).astype("timedelta64[s]").astype(int)
jump_indices = np.where(time_differences > 1)[0] + 1

# Split the data into crossing groups
split_indices = [0] + jump_indices.tolist() + [len(times)]

# We split the data to reform the groups at these points
# Split indices mark the start of each group (except for the last point,
# which doesn't get used as a start point via the zip logic below)
# Negative jump indices are corrected for and should be excluded
split_indices = [0] + jump_indices.tolist() + [len(times)]
split_indices.sort()

# We need some buffer the classifications to remove eroneous edge cases which
# lack sufficient data
buffer = 5
crossing_groups = [
    {
        "Time": times[start:end],
        "P(Solar Wind)": probabilities["Solar Wind"][start:end],
        "P(Magnetosheath)": probabilities["Magnetosheath"][start:end],
        "P(Magnetosphere)": probabilities["Magnetosphere"][start:end],
    }
    for start, end in zip(
        np.add(split_indices[:-1], buffer), np.subtract(split_indices[1:], buffer)
    )
]

# Now we can loop through these crossing groups and apply our logic
groups_without_crossings = 0
new_crossings = []
all_regions = []
for crossing_group in tqdm(
    crossing_groups, desc="Finding Crossings", total=len(crossing_groups)
):

    # Convert to datetime
    # astype("O") converts to python object
    crossing_group["Time"] = crossing_group["Time"].astype("datetime64[ms]").astype("O")

    probabilities = np.vstack(
        [
            crossing_group["P(Magnetosheath)"],
            crossing_group["P(Magnetosphere)"],
            crossing_group["P(Solar Wind)"],
        ]
    ).T

    sorted_probabilities = np.sort(probabilities, axis=1)
    largest_probabilities = sorted_probabilities[:, -1]
    second_largest_probabities = sorted_probabilities[:, -2]

    probability_ratio = second_largest_probabities / largest_probabilities

    highest_probabilities = np.argmax(probabilities, axis=1)

    is_magnetosheath = highest_probabilities == 0
    is_magnetosphere = highest_probabilities == 1
    is_solar_wind = highest_probabilities == 2

    # FIND CROSSINGS
    region_labels = np.empty(len(crossing_group["Time"]), dtype=object)
    region_labels[is_magnetosheath] = "Magnetosheath"
    region_labels[is_magnetosphere] = "Magnetosphere"
    region_labels[is_solar_wind] = "Solar Wind"

    # Define regions
    crossing_indices = np.where(region_labels[:-1] != region_labels[1:])[0]

    # N = 0 is not possible as we centre on a crossing interval
    # N = 1 we can't determine metrics for as region duration is undefined
    if len(crossing_indices) > 1:

        regions = []

        for i in range(len(crossing_indices)):
            current_crossing_index = crossing_indices[i]

            if i == len(crossing_indices) - 1:
                break

            else:
                next_crossing_index = crossing_indices[i + 1]

            regions.append(
                {
                    "Start Time": crossing_group["Time"][current_crossing_index]
                    + dt.timedelta(seconds=1 / 2),
                    "End Time": crossing_group["Time"][next_crossing_index]
                    + dt.timedelta(seconds=1 / 2),
                    "Duration (seconds)": (
                        crossing_group["Time"][next_crossing_index]
                        - crossing_group["Time"][current_crossing_index]
                    ).total_seconds(),
                    "Label": region_labels[current_crossing_index + 1],
                    # Including the values at the crossing points
                    "Confidence": 1
                    - np.median(
                        probability_ratio[
                            current_crossing_index : next_crossing_index + 1
                        ]
                    ),
                }
            )

        # Add in the first and last region
        regions.insert(
            0,
            {
                "Start Time": crossing_group["Time"][0] - dt.timedelta(seconds=1 / 2),
                "End Time": regions[0]["Start Time"],
                "Duration (seconds)": (
                    regions[0]["Start Time"]
                    - (crossing_group["Time"][0] - dt.timedelta(seconds=1 / 2))
                ).total_seconds(),
                "Label": region_labels[0],
                "Confidence": 1,  # Assume good confidence for first and last
            },
        )
        regions.append(
            {
                "Start Time": regions[-1]["End Time"],
                "End Time": crossing_group["Time"][-1] + dt.timedelta(seconds=1 / 2),
                "Duration (seconds)": (
                    crossing_group["Time"][-1]
                    + dt.timedelta(seconds=1 / 2)
                    - regions[-1]["Start Time"]
                ).total_seconds(),
                "Label": region_labels[-1],
                "Confidence": 1,  # Assume good confidence for first and last
            },
        )

    elif len(crossing_indices) == 1:
        regions = []
        # ONLY ONE CHANGE IN REGION
        regions.append(
            {
                "Start Time": crossing_group["Time"][0] - dt.timedelta(seconds=1 / 2),
                "End Time": crossing_group["Time"][crossing_indices[0]]
                + dt.timedelta(seconds=1 / 2),
                "Duration (seconds)": (
                    crossing_group["Time"][crossing_indices[0]]
                    + dt.timedelta(seconds=1 / 2)
                    - (crossing_group["Time"][0] - dt.timedelta(seconds=1 / 2))
                ).total_seconds(),
                "Label": region_labels[crossing_indices[0] - 1],
                # Including the values at the crossing points
                "Confidence": 1
                - np.median(probability_ratio[0 : crossing_indices[0] + 1]),
            }
        )
        regions.append(
            {
                "Start Time": crossing_group["Time"][crossing_indices[0]]
                + dt.timedelta(seconds=1 / 2),
                "End Time": crossing_group["Time"][-1] + dt.timedelta(seconds=1 / 2),
                "Duration (seconds)": (
                    crossing_group["Time"][-1]
                    + dt.timedelta(seconds=1 / 2)
                    - (
                        crossing_group["Time"][crossing_indices[0]]
                        + dt.timedelta(seconds=1 / 2)
                    )
                ).total_seconds(),
                "Label": region_labels[crossing_indices[0] + 1],
                # Including the values at the crossing points
                "Confidence": 1 - np.mean(probability_ratio[crossing_indices[0] : -1]),
            }
        )

    else:
        # print("No crossings detected")
        # print("Skipping...")
        groups_without_crossings += 1
        continue

    region_data = pd.DataFrame(regions)

    region_data.loc[
        ~(
            (region_data["Duration (seconds)"] >= 65)
            | (region_data["Confidence"] > 0.757)
        ),
        "Label",
    ] = "Unknown"

    # Save regions
    all_regions.append(region_data)

    # DETERMINE CROSSINGS
    for region_id, region in region_data.iterrows():

        # We place crossings at the end of regions
        # Need to skip last region

        if region_id == len(region_data) - 1:
            break

        next_region = region_data.loc[region_id + 1]

        transition = ""

        if region["Label"] == "Solar Wind":

            match next_region["Label"]:

                case "Magnetosheath":
                    transition = "BS_IN"

                case "Unknown":
                    transition = "UKN (SW -> UKN)"

                case "Solar Wind":
                    continue

                case "Magnetosphere":
                    transition = "UNPHYS (SW -> MSp)"

                case _:
                    raise ValueError(
                        f"Unknown region transition: Solar Wind -> {next_region['Label']}"
                    )

        elif region["Label"] == "Magnetosheath":

            match next_region["Label"]:

                case "Solar Wind":
                    transition = "BS_OUT"

                case "Magnetosphere":
                    transition = "MP_IN"

                case "Unknown":
                    transition = "UKN (MSh -> UKN)"

                case "Magnetosheath":
                    continue

                case _:
                    raise ValueError(
                        f"Unknown region transition: Magnetosheath -> {next_region['Label']}"
                    )

        elif region["Label"] == "Magnetosphere":

            match next_region["Label"]:

                case "Magnetosheath":
                    transition = "MP_OUT"

                case "Unknown":
                    transition = "UKN (MSp -> UKN)"

                case "Magnetosphere":
                    continue

                case "Solar Wind":
                    transition = "UNPHYS (MSp -> SW)"

                case _:
                    raise ValueError(
                        f"Unknown region transition: Magnetosphere -> {next_region['Label']}"
                    )

        elif region["Label"] == "Unknown":

            match next_region["Label"]:

                case "Solar Wind":
                    transition = "UKN (UKN -> SW)"

                case "Magnetosheath":
                    transition = "UKN (UKN -> MSh)"

                case "Magnetosphere":
                    transition = "UKN (UKN -> MSp)"

                case "Unknown":
                    continue

                case _:
                    raise ValueError(
                        f"Unknown region transition: Unknown -> {next_region['Label']}"
                    )

        else:
            raise ValueError(f"Unknown region label: {region['Label']}")

        new_crossing = {
            "Time": region["End Time"],
            "Transition": transition,
            "Confidence": (region["Confidence"] + next_region["Confidence"]) / 2,
        }

        new_crossings.append(new_crossing)

print(f"Number of crossings detected: {len(new_crossings)}")
print(
    f"Number of crossing interval groups without detections: {groups_without_crossings}"
)

# Count occurrences of each transition type
transition_counts = collections.Counter(
    crossing["Transition"] for crossing in new_crossings
)

# Print results
for transition, count in transition_counts.items():
    print(f"{transition}: {count} crossings")


pd.DataFrame(new_crossings).to_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings.csv"
)
pd.concat(all_regions).to_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_regions.csv"
)

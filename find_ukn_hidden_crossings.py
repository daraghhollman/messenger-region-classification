"""
Script to identify crossings from one region to another which are not marked as crossings due to
transitioning through a unknown region.
"""

import collections

import numpy as np
import pandas as pd

# Load the new crossing list
hollman_crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings.csv"
)
hollman_crossings["Time"] = pd.to_datetime(hollman_crossings["Time"])

new_crossings = []
for i, current_crossing in hollman_crossings.iterrows():

    if i == len(hollman_crossings) - 1:
        break

    next_crossing = hollman_crossings.loc[i + 1]

    match current_crossing["Transition"]:

        case "UKN (SW -> UKN)":
            if next_crossing["Transition"] == "UKN (UKN -> MSh)":
                new_transition = "BS_IN"

            else:
                continue

        case "UKN (MSh -> UKN)":
            if next_crossing["Transition"] == "UKN (UKN -> SW)":
                new_transition = "BS_OUT"

            elif next_crossing["Transition"] == "UKN (UKN -> MSp)":
                new_transition = "MP_IN"

            else:
                continue

        case "UKN (MSp -> UKN)":
            if next_crossing["Transition"] == "UKN (UKN -> MSh)":
                new_transition = "MP_OUT"

            else:
                continue

        case _:
            continue

    new_crossings.append(
        {
            "Transition": new_transition,
            "Time": current_crossing["Time"]
            + (next_crossing["Time"] - current_crossing["Time"]) / 2,
            "Confidence": np.nan,
        }
    )


new_crossings = pd.DataFrame(new_crossings)
new_hollman_crossings = pd.concat([hollman_crossings, new_crossings])


# Count occurrences of each transition type
transition_counts = collections.Counter(
    c for c in new_hollman_crossings["Transition"] if "UKN" not in c
)

# Print results
for transition, count in transition_counts.items():
    print(f"{transition}: {count} crossings")

new_hollman_crossings.to_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings_with_adjusted_counting.csv"
    if not no_heliocentric_distance
    else "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings_without_heliocentric_distance_with_adjusted_counting.csv"
)

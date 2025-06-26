"""
A post processing step to add a new crossing at the middle of unknown regions which hide a region transition
Unknown crossings are ignored
"""

import collections

import pandas as pd


def main():
    # Load the post-processed regions
    model_regions = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/post-processing/1_bookend_regions_processed.csv",
        parse_dates=["Start Time", "End Time"],
    )

    new_crossing_times = []
    new_crossing_labels = []

    region_transition_map = {
        "Solar Wind": {
            "Magnetosheath": "BS_IN",
            "Magnetosphere": "UNPHYSICAL (SW -> MSp)",
        },
        "Magnetosheath": {
            "Solar Wind": "BS_OUT",
            "Magnetosphere": "MP_IN",
        },
        "Magnetosphere": {
            "Magnetosheath": "MP_OUT",
            "Solar Wind": "UNPHYSICAL (MSp -> SW)",
        },
    }

    # Loop through all regions and stop at any unknown regions
    for i, region in model_regions.iterrows():

        if i == 0:
            continue

        previous_region = model_regions.iloc[i - 1]

        # If both the currrent region and the previous region are not unknown,
        # we might want to place a crossing there.
        if (region["Label"] != "Unknown") and (previous_region["Label"] != "Unknown"):
            # We still want the crossings associated with this region
            # To avoid placing crossings between applications, we only place
            # crossings where the start and end times of adjacent regions are
            # equal.
            if region["Start Time"] == previous_region["End Time"]:
                new_crossing_times.append(region["Start Time"])
                new_crossing_labels.append(
                    region_transition_map[previous_region["Label"]][region["Label"]]
                )

            continue

        # Possiblities remaining are:
        #   - previous region: unknown
        #   - current region: unknown
        # Both unknown is not possible due to previous combining of regions.

        # If the previous region is unknown, then transition to the current
        # region has already been handled and we can continue.
        if previous_region["Label"] == "Unknown":
            continue

        # Check region before
        label_before = previous_region["Label"]
        label_after = model_regions.loc[i + 1]["Label"]

        # If the before and after labels are different, at least one crossing must
        # have occured within the unknown region. We assume this occurs at the
        # midpoint of the unknown region.
        if label_before == label_after:
            continue

        mid_time = (
            region["Start Time"] + (region["End Time"] - region["Start Time"]) / 2
        )
        new_crossing_times.append(mid_time)
        new_crossing_labels.append(region_transition_map[label_before][label_after])

    new_crossings = pd.DataFrame(
        {"Times": new_crossing_times, "Label": new_crossing_labels}
    )

    # Count occurrences of each crossing type
    transition_counts = collections.Counter(new_crossings["Label"])

    # Print results
    for transition, count in transition_counts.items():
        print(f"{transition}: {count} crossings")

    new_crossings.to_csv("./zenodo-data/hollman_2025_crossing_list.csv")


if __name__ == "__main__":
    main()

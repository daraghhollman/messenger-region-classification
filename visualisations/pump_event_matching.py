"""
Script to find what percent of crossings from Pump+ 2024 (dataset, https://zenodo.org/records/14051199)
are found in our new crossing list.

This list by Kristin Pump uses an automated threshold-based method to find fast shocks in Philpott+ 2020
boundary intervals. The columns are as follows:

['time', 'orbit_number', 'x_mso_km', 'y_mso_km', 'z_mso_km', 'in/out', 'jump_ratio', 'indicator']

with indicator acting as a quality code.
    1 - Very clear crossings, very high quality
    2 - Clear crossings, good quality
    3 - Unclear crossings, poor quality (Not recommended for use)

Temporal uncertainty is +/- 2 seconds
"""

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# Load the new crossing list
hollman_crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings.csv"
)
hollman_crossings["Time"] = pd.to_datetime(hollman_crossings["Time"])

# Load the Pump crossings
pump_crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/pump_bowshock_bdry_detection_v2.csv"
)
pump_crossings["datetime"] = pd.to_datetime(pump_crossings["datetime"])
pump_crossings["Time"] = pump_crossings["datetime"]


def Match_Events(event_list_a: pd.DataFrame, event_list_b: pd.DataFrame, max_temporal_difference: float | None):
    """
    Finds the nearest match for each element in `event_list_a` within `max_temporal_difference` (seconds)
    of elements in `event_list_b`. Returns the indices of closest match in list B along with the temporal
    difference. If no match is found, no index or temporal difference is recorded.

    Assumes both dataframes have column `Time` of type datetime.datetime
    """

    # Make sure lists are sorted by time
    event_list_a = event_list_a.sort_values("Time").reset_index(drop=True)
    event_list_b = event_list_b.sort_values("Time").reset_index(drop=True)

    b_times = event_list_b["Time"]
    b_matches = []

    for i, row in tqdm(event_list_a.iterrows(), total=len(event_list_a)):

        # Find the time differences in seconds
        time_differences = np.abs((b_times - row["Time"]) / dt.timedelta(seconds=1))

        # Find the index with the smallest time difference
        matching_index = np.argmin(time_differences)
        matching_time_difference = time_differences[matching_index]

        # Test if within threshold
        if max_temporal_difference is not None:
            if matching_time_difference <= max_temporal_difference:
                b_matches.append({"Index A": i, "Index B": matching_index, "Time Difference": matching_time_difference})

            else:
                b_matches.append({"Index A": i, "Index B": np.nan, "Time Difference": np.nan})

        else:
            b_matches.append({"Index A": i, "Index B": matching_index, "Time Difference": matching_time_difference})


    return pd.DataFrame(b_matches)


"""
print("All Pump events")
matched_events = Match_Events(pump_crossings, hollman_crossings, max_temporal_difference=10)

valid_events = matched_events.dropna()

print(len(valid_events) / len(matched_events))
print(valid_events)
"""

t = 10

print("\n")
print("High quality")
matched_events = Match_Events(pump_crossings.loc[pump_crossings["indicator"] == 1], hollman_crossings, max_temporal_difference=t)

valid_events = matched_events.dropna()

print(len(valid_events) / len(matched_events))
print(valid_events)

print("\n")
print("Good quality")
matched_events = Match_Events(pump_crossings.loc[pump_crossings["indicator"] == 2], hollman_crossings, max_temporal_difference=t)

valid_events = matched_events.dropna()

print(len(valid_events) / len(matched_events))
print(valid_events)

print("\n")
print("Poor quality")
matched_events = Match_Events(pump_crossings.loc[pump_crossings["indicator"] == 3], hollman_crossings, max_temporal_difference=t)

valid_events = matched_events.dropna()

print(len(valid_events) / len(matched_events))
print(valid_events)
"""

match_ratios = []
times_to_check = [1, 2, 6, 10, 20, 40, 60]
for t in times_to_check:

    hq_matched_events = Match_Events(pump_crossings.loc[pump_crossings["indicator"] == 1], hollman_crossings, max_temporal_difference=t)
    hq_match_ratio = len(hq_matched_events.dropna()) / len(hq_matched_events)
    
    gq_matched_events = Match_Events(pump_crossings.loc[pump_crossings["indicator"] == 2], hollman_crossings, max_temporal_difference=t)
    gq_match_ratio = len(gq_matched_events.dropna()) / len(gq_matched_events)

    pq_matched_events = Match_Events(pump_crossings.loc[pump_crossings["indicator"] == 3], hollman_crossings, max_temporal_difference=t)
    pq_match_ratio = len(pq_matched_events.dropna()) / len(pq_matched_events)

    match_ratios.append([hq_match_ratio, gq_match_ratio, pq_match_ratio])

match_ratios = np.array(match_ratios)

fig, ax = plt.subplots()

ax.plot(times_to_check, match_ratios[:,0], label="High Quality")
ax.plot(times_to_check, match_ratios[:,1], label="Good Quality")
ax.plot(times_to_check, match_ratios[:,2], label="Poor Quality")

ax.set_xlabel("Max Time Difference")
ax.set_ylabel("Matching Ratio")
ax.margins(0)
ax.legend()

plt.show()
"""

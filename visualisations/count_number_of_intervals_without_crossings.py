import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from hermpy import boundaries, utils

# Load crossing list and intervals list
intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False
)
new_crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/hollman_2025_crossing_list.csv"
)
new_crossings["Time"] = pd.to_datetime(new_crossings["Times"])

# For each interval, count the number of crossings within it. If the number is
# 0, add this to a counter
no_crossings_count = 0
missing_crossing_times = []

tolerance = dt.timedelta(seconds=1000)

for _, interval in intervals.iterrows():

    crossings_within_interval = (
        new_crossings["Time"]
        .between(interval["Start Time"] - tolerance, interval["End Time"] + tolerance)
        .sum()
    )

    if crossings_within_interval == 0:
        no_crossings_count += 1
        missing_crossing_times.append(interval["Start Time"])

print(no_crossings_count)
print(no_crossings_count / len(intervals))

plt.hist(missing_crossing_times)

plt.show()

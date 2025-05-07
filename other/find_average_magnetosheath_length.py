"""
Find the ammount of time between BS_IN and MP_IN as well as MP_OUT and BP_OUT
"""

import numpy as np
from hermpy import boundaries, utils

philpott_intervals = boundaries.Load_Crossings(utils.User.CROSSING_LISTS["Philpott"])

magnetosheath_lengths = []

for i, interval in philpott_intervals.iterrows():

    if i == len(philpott_intervals) - 1:
        break

    next_interval = philpott_intervals.loc[i + 1]

    if interval["Type"] == "BS_IN":
        if next_interval["Type"] == "MP_IN":
            magnetosheath_lengths.append(
                next_interval["Start Time"] - interval["End Time"]
            )

    elif interval["Type"] == "MP_OUT":
        if next_interval["Type"] == "BS_OUT":
            magnetosheath_lengths.append(
                next_interval["Start Time"] - interval["End Time"]
            )

    else:
        continue


print(np.mean(magnetosheath_lengths))

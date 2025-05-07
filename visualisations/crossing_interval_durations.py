"""
Script to see how the duration of crossing intervals changes with heliocentric distance
"""

import matplotlib.pyplot as plt
import numpy as np
from hermpy import boundaries, trajectory, utils

only_of_type = "MP"  # "", "BS", "MP"

# Load Philpott intervals
philpott_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False, backend="Philpott"
)

# Load Sun intervals
sun_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Sun"], include_data_gaps=False, backend="Sun"
)

if only_of_type != "":
    philpott_intervals = philpott_intervals.loc[
        philpott_intervals["Type"].str.contains(only_of_type)
    ]
    sun_intervals = sun_intervals.loc[sun_intervals["Type"].str.contains(only_of_type)]

philpott_duration = (
    philpott_intervals["End Time"] - philpott_intervals["Start Time"]
).dt.total_seconds()
sun_duration = (
    sun_intervals["End Time"] - sun_intervals["Start Time"]
).dt.total_seconds()


# print(sum(sun_duration.to_numpy() < 0))
# Some events in the Sun list are of negative duration due to errors in the
# creation of the list. We are just interested in the distribution here, and
# can just remove these.
sun_duration = sun_duration.loc[sun_duration > 0]

philpott_intervals["Duration"] = philpott_duration
sun_intervals["Duration"] = sun_duration

fig, axes = plt.subplots(1, 2)

philpott_axis, sun_axis = axes

# Get heliocentric distance and plot
for intervals in [philpott_intervals, sun_intervals]:

    intervals["Mid Time"] = (
        intervals["Start Time"] + (intervals["End Time"] - intervals["Start Time"]) / 2
    )
    intervals["Heliocentric Distance (AU)"] = utils.Constants.KM_TO_AU(
        trajectory.Get_Heliocentric_Distance(intervals["Mid Time"])
    )

philpott_intervals = philpott_intervals.dropna()
sun_intervals = sun_intervals.dropna()

bin_size = 0.01
heliocentric_distance_bins = np.arange(0.3, 0.47 + bin_size, bin_size)

duration_bin_size = 50
duration_bins = np.arange(0, 1000 + duration_bin_size, duration_bin_size)

residence, _ = np.histogram(
    philpott_intervals["Heliocentric Distance (AU)"],
    bins=heliocentric_distance_bins,
    density=True,
)

duration_distribution, _ = np.histogram(
    philpott_intervals["Duration"],
    bins=duration_bins,
    density=True,
)

for intervals, ax in zip([philpott_intervals, sun_intervals], axes):
    h, heliocentric_distance_edges, duration_edges = np.histogram2d(
        intervals["Heliocentric Distance (AU)"],
        intervals["Duration"],
        bins=[heliocentric_distance_bins, duration_bins],
    )

    h /= residence[:, np.newaxis]

    pcm = ax.pcolormesh(heliocentric_distance_edges, duration_bins, h.T)

    ax.set_xlabel("Heliocentric Distance [AU]")
    ax.set_ylabel("Interval Duration [seconds]")

    plt.colorbar(
        pcm, label="# Intervals, normalised by residence in heliocentric distance"
    )

if only_of_type != "":
    philpott_axis.set_title(f"Philpott Intervals ({only_of_type} only)")
    sun_axis.set_title(f"Sun Intervals ({only_of_type} only)")


else:
    philpott_axis.set_title("Philpott Intervals")
    sun_axis.set_title("Sun Intervals")

plt.show()

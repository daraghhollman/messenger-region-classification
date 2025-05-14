"""
Script to see how pump crossings vary with heliocentric distance
A simple histogram of counts
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy import boundaries, trajectory, utils
from hermpy.plotting import wong_colours

# Load philpott crossings as a proxy for residence
philpott_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False
)
philpott_intervals["Mid Time"] = (
    philpott_intervals["Start Time"]
    + (philpott_intervals["End Time"] - philpott_intervals["Start Time"]) / 2
)
philpott_intervals["Heliocentric Distance"] = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(philpott_intervals["Mid Time"])
)

# Load pump crossings
pump_crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/pump_bowshock_bdry_detection_v2.csv"
)
print(pump_crossings["datetime"])
pump_crossings["datetime"] = pd.to_datetime(pump_crossings["datetime"])

# Find heliocentric distance
pump_crossings["Heliocentric Distance"] = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(pump_crossings["datetime"])
)

bin_size = 0.01  # AU
bin_edges = np.arange(0.3, 0.47 + bin_size, bin_size)

fig, ax = plt.subplots()

indicator_3, _, _ = ax.hist(
    pump_crossings.loc[pump_crossings["indicator"] == 3]["Heliocentric Distance"],
    bins=bin_edges.tolist(),
    histtype="step",
    color=wong_colours["yellow"],
    label="Indicator 3",
    linewidth=3,
    density=True,
)
indicator_2, _, _ = ax.hist(
    pump_crossings.loc[pump_crossings["indicator"] == 2]["Heliocentric Distance"],
    bins=bin_edges.tolist(),
    histtype="step",
    color=wong_colours["blue"],
    label="Indicator 2",
    linewidth=3,
    density=True,
)
indicator_1, _, _ = ax.hist(
    pump_crossings.loc[pump_crossings["indicator"] == 1]["Heliocentric Distance"],
    bins=bin_edges.tolist(),
    histtype="step",
    color=wong_colours["green"],
    linewidth=3,
    label="Indicator 1",
    density=True,
)

philpott_histogram, _, _ = ax.hist(
    philpott_intervals["Heliocentric Distance"],
    bins=bin_edges.tolist(),
    color=wong_colours["red"],
    histtype="step",
    linewidth=5,
    label="Philpott Intervals (BS & MP)",
    density=True,
)

# all
pump_histogram, _, _ = ax.hist(
    pump_crossings["Heliocentric Distance"],
    bins=bin_edges.tolist(),
    histtype="step",
    color="black",
    label="Pump Bow Shock Crossings",
    linewidth=5,
    density=True,
    zorder=5,
)


ax.legend()

ax.set_xlabel("Heliocentric Distance (AU)")
ax.set_ylabel("Crossing Density\n(Crossings / (N * bin width) )")

# DIVISION PLOT
fig, ax = plt.subplots()

bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

ax.scatter(
    bin_centres,
    pump_histogram / philpott_histogram,
    c="k",
    marker=".",
)
ax.scatter(
    bin_centres,
    indicator_3 / philpott_histogram,
    c="yellow",
    marker=".",
)
ax.scatter(
    bin_centres,
    indicator_2 / philpott_histogram,
    c="blue",
    marker=".",
)
ax.scatter(
    bin_centres,
    indicator_1 / philpott_histogram,
    c="green",
    marker=".",
)

ax.set_facecolor("lightgrey")
ax.set_xlabel("Heliocentric Distance (AU)")
ax.set_ylabel("Crossing Denstiy / Philpott Interval Density\n(Black / Orange)")

plt.show()

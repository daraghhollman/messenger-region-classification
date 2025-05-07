"""
Scatter region confidence against heliocentric distance
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy import boundaries, trajectory, utils

# Load Philpott intervals to get Heliocentric residence
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

# Load regions
regions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_regions.csv"
)
regions["Start Time"] = pd.to_datetime(regions["Start Time"])
regions["End Time"] = pd.to_datetime(regions["End Time"])

# Find heliocentric distance for midpoint of region
regions["Mid Time"] = (
    regions["Start Time"] + (regions["End Time"] - regions["Start Time"]) / 2
)

regions["Heliocentric Distance"] = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(regions["Mid Time"])
)


bin_size = 0.01
heliocentric_distance_bins = np.arange(0.3, 0.47 + bin_size, bin_size)

confidence_bin_size = 0.1
confidence_bins = np.arange(0, 1 + confidence_bin_size, confidence_bin_size)


residence, _ = np.histogram(
    philpott_intervals["Heliocentric Distance"],
    bins=heliocentric_distance_bins,
    density=True,
)

confidence_distribution, _ = np.histogram(
    regions["Confidence"],
    bins=confidence_bins,
    density=True,
)

h, heliocentric_distance_edges, confidence_edges = np.histogram2d(
    regions["Heliocentric Distance"],
    regions["Confidence"],
    bins=[heliocentric_distance_bins, confidence_bins],
)

# Normalise
normalise = True
if normalise:

    # Outer product of underlying distributions
    outer_product = np.outer(residence, confidence_distribution)

    h /= outer_product

fig, ax = plt.subplots()

pcm = ax.pcolormesh(heliocentric_distance_edges, confidence_edges, h.T)

ax.set_xlabel("Heliocentric Distance [AU]")
ax.set_ylabel("Region Confidence")

plt.colorbar(pcm, label="# Regions" if not normalise else "Normalised # Regions")

plt.show()


# Invstigate region length
regions["Duration"] = regions["End Time"] - regions["Start Time"]

fig, ax = plt.subplots()

ax.scatter(regions["Heliocentric Distance"], regions["Duration"].dt.total_seconds())

ax.set_xlabel("Heliocentric Distance [AU]")
ax.set_ylabel("Region Duration")

plt.show()

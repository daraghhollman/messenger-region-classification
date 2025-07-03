"""
Script to investigate count of BS and MP with respect to heliocentric distance
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from hermpy import boundaries, trajectory, utils

wong_colours = {
    "black": "black",
    "orange": "#E69F00",
    "light blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "pink": "#CC79A7",
}


# Load crossings
crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/hollman_2025_crossing_list.csv"
)
crossings["Time"] = pd.to_datetime(crossings["Times"])
crossings["Transition"] = crossings["Label"]

philpott_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False
)

bow_shock_crossings = crossings.loc[crossings["Transition"].str.contains("BS")].copy()
magnetopause_crossings = crossings.loc[
    crossings["Transition"].str.contains("MP")
].copy()

# Get heliocentric distances
bow_shock_crossings["Heliocentric Distance"] = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(bow_shock_crossings["Time"])
)
magnetopause_crossings["Heliocentric Distance"] = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(magnetopause_crossings["Time"])
)

philpott_intervals["Mid Time"] = (
    philpott_intervals["Start Time"]
    + (philpott_intervals["End Time"] - philpott_intervals["Start Time"]) / 2
)
philpott_intervals["Heliocentric Distance"] = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(philpott_intervals["Mid Time"])
)

bin_size = 0.01
heliocentric_distance_bins = np.arange(0.3, 0.47 + bin_size, bin_size)

fig, ax = plt.subplots()

use_density = True

b_data, _, _ = ax.hist(
    bow_shock_crossings["Heliocentric Distance"],
    bins=heliocentric_distance_bins.tolist(),
    color="black",
    histtype="step",
    linewidth=5,
    label="Bow Shock Crossings",
    density=use_density,
)

m_data, _, _ = ax.hist(
    magnetopause_crossings["Heliocentric Distance"],
    bins=heliocentric_distance_bins.tolist(),
    color=wong_colours["light blue"],
    histtype="step",
    linewidth=5,
    label="Magnetopause Crossings",
    density=use_density,
)

p_data, _, _ = ax.hist(
    philpott_intervals["Heliocentric Distance"],
    bins=heliocentric_distance_bins.tolist(),
    color=wong_colours["red"],
    histtype="step",
    linewidth=5,
    label="Philpott Intervals (BS & MP)",
    density=use_density,
)

ax.set_xlabel("Heliocentric Distance (AU)")

if not use_density:
    ax.set_ylabel("Number of Crossings")

else:
    ax.set_ylabel("Crossing Density\n(Crossings / (N * bin width) )")

ax.legend()
ax.margins(0)

plt.show()

fig, ax = plt.subplots()

bin_centres = (heliocentric_distance_bins[1:] + heliocentric_distance_bins[:-1]) / 2

b_pearsonr = scipy.stats.pearsonr(bin_centres, b_data / p_data)
m_pearsonr = scipy.stats.pearsonr(bin_centres, m_data / p_data)

ax.scatter(
    bin_centres,
    b_data / p_data,
    c="k",
    label=f"Bow Shock: r={b_pearsonr.statistic:.2f}, p={b_pearsonr.pvalue:.10f}",
)
ax.scatter(
    bin_centres,
    m_data / p_data,
    c=wong_colours["light blue"],
    label=f"Magnetopause: r={m_pearsonr.statistic:.2f}, p={m_pearsonr.pvalue:.5f}",
)

ax.legend()
ax.set_xlabel("Heliocentric Distance (AU)")
ax.set_ylabel("Crossing Denstiy / Philpott Interval Density\n(Black / Orange)")


plt.show()

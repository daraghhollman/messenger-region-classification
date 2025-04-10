"""
Script to investigate count of BS and MP with respect to heliocentric distance
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy import trajectory, utils

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
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings.csv"
)
crossings["Time"] = pd.to_datetime(crossings["Time"])

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

bin_size = 0.01
heliocentric_distance_bins = np.arange(0.3, 0.47 + bin_size, bin_size)

fig, ax = plt.subplots()

ax.hist(
    bow_shock_crossings["Heliocentric Distance"],
    bins=heliocentric_distance_bins.tolist(),
    color="black",
    histtype="step",
    linewidth=5,
    label="Bow Shock Crossings"
)

ax.hist(
    magnetopause_crossings["Heliocentric Distance"],
    bins=heliocentric_distance_bins.tolist(),
    color=wong_colours["light blue"],
    histtype="step",
    linewidth=5,
    label="Magnetopause Crossings"
)

ax.set_xlabel("Heliocentric Distance (AU)")
ax.set_ylabel("Number of Crossings")

ax.legend()
ax.margins(0)

plt.show()

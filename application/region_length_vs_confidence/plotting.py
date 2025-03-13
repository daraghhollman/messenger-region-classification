"""
A script to plot data from get_points.py, to try find a physically reasonable minimum crossing duration.

Region confidence is described by:

    1 - Mean( Probability Ratio within the region )

where:

    Probability Ratio = second largest probability / largest probability

for each classification within the region. i.e. each second.
"""

import matplotlib.pyplot as plt
import pandas as pd

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

# Load data
region_metrics = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/application/region_length_vs_confidence/region_metrics.csv"
)

fig, ax = plt.subplots()

ax.scatter(
    region_metrics["Duration (seconds)"],
    region_metrics["Confidence (Mean Method)"],
    marker=".",
    color="black",
    label=f"N={len(region_metrics)}",
)

ax.set_xlabel("Region Duration [seconds]")
ax.set_ylabel("Region Confidence (Mean Method) [arb.]")

ax.margins(0)

ax.legend()

ax.set_xlim(0, 100)

"""

fig, ax = plt.subplots()

ax.scatter(
    region_metrics["Duration (seconds)"],
    region_metrics["Confidence (Median Method)"],
    marker=".",
    color="black",
    label=f"N={len(region_metrics)}",
)

ax.set_xlabel("Region Duration [seconds]")
ax.set_ylabel("Region Confidence (Median Method) [arb.]")

ax.legend()

ax.margins(0)

"""
plt.show()

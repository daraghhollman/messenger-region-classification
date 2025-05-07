import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from hermpy import boundaries, trajectory, utils
from hermpy.plotting import wong_colours

# Load the regions
new_regions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_regions.csv"
)

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


unknown_regions = new_regions.loc[new_regions["Label"] == "Unknown"]

unknown_regions["Start Time"] = pd.to_datetime(unknown_regions["Start Time"])
unknown_regions["End Time"] = pd.to_datetime(unknown_regions["End Time"])

# Get heliocentric distance
unknown_regions["Mid Time"] = (
    unknown_regions["Start Time"]
    + (unknown_regions["End Time"] - unknown_regions["Start Time"]) / 2
)
unknown_regions["Heliocentric Distance (AU)"] = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(unknown_regions["Mid Time"])
)

fig, ax = plt.subplots()

bin_size = 0.01
heliocentric_distance_bins = np.arange(0.3, 0.47 + bin_size, bin_size)

unknown_region_data, _, _ = ax.hist(
    unknown_regions["Heliocentric Distance (AU)"],
    bins=heliocentric_distance_bins.tolist(),
    histtype="step",
    color="black",
    linewidth=5,
    density=True,
    label="Unknown regions",
)

residence_data, _, _ = ax.hist(
    philpott_intervals["Heliocentric Distance"],
    bins=heliocentric_distance_bins.tolist(),
    color=wong_colours["red"],
    histtype="step",
    linewidth=5,
    label="Philpott Intervals (BS & MP)",
    density=True,
)

ax.legend()
ax.margins(x=0)

ax.set_xlabel("Heliocentric Distance [AU]")
ax.set_ylabel("Area-normalised Occurance")

plt.show()

fig, ax = plt.subplots()

bin_centres = (heliocentric_distance_bins[:-1] + heliocentric_distance_bins[1:]) / 2

pearsonr = scipy.stats.pearsonr(bin_centres, unknown_region_data / residence_data)

ax.scatter(
    bin_centres,
    unknown_region_data / residence_data,
    c="k",
    label=f"Normalised distribution of unknown regions\nr={pearsonr.statistic:.3f}, p={pearsonr.pvalue:.3f}",
)

ax.set_xlabel("Heliocentric Distance [AU]")
ax.set_ylabel("Normalized Occurance")

ax.legend()


plt.show()

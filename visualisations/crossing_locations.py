import numpy as np
import pandas as pd
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt

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
crossings = pd.read_csv("/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings.csv")
crossings["Time"] = pd.to_datetime(crossings["Time"])

bow_shock_crossings = crossings.loc[crossings["Transition"].str.contains("BS")].copy()
magnetopause_crossings = crossings.loc[crossings["Transition"].str.contains("MP")].copy()

bow_shock_locations = trajectory.Get_Position("MESSENGER", bow_shock_crossings["Time"], frame="MSM", aberrate="average")
magnetopause_locations = trajectory.Get_Position("MESSENGER", magnetopause_crossings["Time"], frame="MSM", aberrate="average")

bow_shock_crossings["Radial Distance"] = np.sqrt(np.sum(bow_shock_locations**2, axis=1)) / utils.Constants.MERCURY_RADIUS_KM
magnetopause_crossings["Radial Distance"] = np.sqrt(np.sum(magnetopause_locations**2, axis=1)) / utils.Constants.MERCURY_RADIUS_KM

bow_shock_long = 180 / np.pi * np.arctan2(bow_shock_locations[:,1], bow_shock_locations[:,0])
bow_shock_long = np.array([x if x > 0 else x + 360 for x in bow_shock_long])
bow_shock_crossings["Local Time"] = ((bow_shock_long + 180) * 24 / 360) % 24

magnetopause_long = 180 / np.pi * np.arctan2(magnetopause_locations[:,1], magnetopause_locations[:,0])
magnetopause_long = np.array([x if x > 0 else x + 360 for x in magnetopause_long])
magnetopause_crossings["Local Time"] = ((magnetopause_long + 180) * 24 / 360) % 24

bow_shock_crossings["Latitude"] = 180 / np.pi * np.arctan2(bow_shock_locations[:,2], np.sqrt(bow_shock_locations[:,0] ** 2 + bow_shock_locations[:,1]))

magnetopause_crossings["Latitude"] = 180 / np.pi * np.arctan2(magnetopause_locations[:,2], np.sqrt(magnetopause_locations[:,0] ** 2 + magnetopause_locations[:,1]))


bow_shock_vars = [bow_shock_crossings[key] for key in ["Radial Distance", "Local Time", "Latitude"]]
magnetopause_vars = [magnetopause_crossings[key] for key in ["Radial Distance", "Local Time", "Latitude"]]

var_labels = ["Radial Distance [Mercury Radii]", "Local Time", "Latitude [degrees]"]

for boundary_vars, name in zip([bow_shock_vars, magnetopause_vars], ["Bow Shock", "Magnetopause"]):

    fig, axes = plt.subplots(1, 3, sharey=True)
    for ax, var, x_label in zip(axes, boundary_vars, var_labels):

        ax.hist(var, label=f"N={len(var)}")
        ax.set_xlabel(x_label)

        ax.margins(0)

        if ax == axes[0]:
            ax.set_ylabel("Number of Crossings")

        ax.legend()
    fig.suptitle(name)

    plt.show()


# Now do this again, but filtering by confidence

# First plot the distriubtion of confidence
fig, ax = plt.subplots()

confidence_bin_size = 0.1
confidence_bins = np.arange(0, 1 + confidence_bin_size, confidence_bin_size).tolist()

ax.hist(
    bow_shock_crossings["Confidence"],
    histtype="step",
    bins=confidence_bins,
    color="black",
    label=f"Bow Shock, N={len(bow_shock_crossings)}",
    density=True,
    alpha=0.8,
    linewidth=5
)

ax.hist(
    magnetopause_crossings["Confidence"],
    histtype="step",
    bins=confidence_bins,
    color=wong_colours["light blue"],
    label=f"Magnetopause, N={len(magnetopause_crossings)}",
    density=True,
    alpha=0.8,
    linewidth=5
)

ax.set_xlabel("Crossing Confidence\n(average neighbouring region confidence)")
ax.set_ylabel("Crossing Density")
ax.legend()

plt.show()


# Now we look at the first distributions in each of the confidence bins
cmap = plt.get_cmap("viridis")
norm = matplotlib.colors.Normalize(0, 1)
scalar_map = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)  # Create mappable for colorbar
for crossings, name in zip([bow_shock_crossings, magnetopause_crossings], ["Bow Shock", "Magnetopause"]):

    fig, axes = plt.subplots(1, 3, sharey=True)

    for i, (bin_start, bin_end) in enumerate(zip(confidence_bins[:-1], confidence_bins[1:])):

        filtered_crossings = crossings.loc[(crossings["Confidence"] >= bin_start) & (crossings["Confidence"] < bin_end)]

        for ax, var in zip(axes, ["Radial Distance", "Local Time", "Latitude"]):
            ax.hist(filtered_crossings[var], color=cmap(i / (len(confidence_bins) - 1)), histtype="step", label=f"{bin_start} <= Crossing Confidence < {bin_end}", linewidth=3)

            ax.set_xlabel(var)
            ax.margins(0)
            # ax.set_yscale("log")

        axes[0].set_ylabel("Number of Crossings")
        axes[1].set_title(name)

    cbar = fig.colorbar(scalar_map, ax=axes, orientation="vertical")
    cbar.set_label("Crossing Confidence")





plt.show()

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy import mag, plotting, trajectory, utils

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

# Find the position of each crossing
# Load full mission data
full_mission = mag.Load_Mission("/home/daraghhollman/Main/data/mercury/messenger_mag")

# Add on the columns of full_mission for the rows in crossings
# Does this using the nearest element
# Its so fast omg
crossings = pd.merge_asof(
    crossings, full_mission, left_on="Time", right_on="date", direction="nearest"
)

bow_shock_crossings = crossings.loc[crossings["Transition"].str.contains("BS")].copy()
magnetopause_crossings = crossings.loc[
    crossings["Transition"].str.contains("MP")
].copy()

# To normalise these distributions by residence, we need the ammount of time spent in each bin.
# Load full mission to get trajectory
positions = [
    full_mission["X MSM' (radii)"],
    full_mission["Y MSM' (radii)"],
    full_mission["Z MSM' (radii)"],
]

bin_size = 0.5
x_bins = np.arange(-5, 5 + bin_size, bin_size)
y_bins = np.arange(-5, 5 + bin_size, bin_size)
z_bins = np.arange(-8, 2 + bin_size, bin_size)
cyl_bins = np.arange(0, 10 + bin_size, bin_size)

# Get residence histograms. These are the frequency of data points. We have
# loaded 1 second average data.
residence_xy, _, _ = np.histogram2d(positions[0], positions[1], bins=[x_bins, y_bins])
residence_xz, _, _ = np.histogram2d(positions[0], positions[2], bins=[x_bins, z_bins])
residence_cyl, _, _ = np.histogram2d(
    positions[0],
    np.sqrt(positions[1] ** 2 + positions[2] ** 2),
    bins=[x_bins, cyl_bins],
)

fig, axes = plt.subplots(2, 3, figsize=(10, 7))

bow_shock_axes = axes[0]
magnetopause_axes = axes[1]

for i, (axes, positions) in enumerate(
    zip(
        [bow_shock_axes, magnetopause_axes],
        [bow_shock_crossings, magnetopause_crossings],
    )
):

    xy_axis, xz_axis, cyl_axis = axes

    xy_hist_data, _, _ = np.histogram2d(
        positions["X MSM' (radii)"], positions["Y MSM' (radii)"], bins=[x_bins, y_bins]
    )
    xz_hist_data, _, _ = np.histogram2d(
        positions["X MSM' (radii)"], positions["Z MSM' (radii)"], bins=[x_bins, z_bins]
    )
    cyl_hist_data, _, _ = np.histogram2d(
        positions["X MSM' (radii)"],
        np.sqrt(positions["Y MSM' (radii)"] ** 2 + positions["Z MSM' (radii)"] ** 2),
        bins=[x_bins, cyl_bins],
    )

    # Normalise
    # Yielding crossings per second

    xy_hist_data = np.where(residence_xy != 0, xy_hist_data / residence_xy, np.nan)
    xz_hist_data = np.where(residence_xz != 0, xz_hist_data / residence_xz, np.nan)
    cyl_hist_data = np.where(residence_cyl != 0, cyl_hist_data / residence_cyl, np.nan)

    # Multiply by 3600 to get crossings per hour
    xy_hist_data *= 3600
    xz_hist_data *= 3600
    cyl_hist_data *= 3600

    # Determine the global vmin and vmax
    vmin, vmax = np.nanmin(
        [
            np.min(xy_hist_data[xy_hist_data > 0]),
            np.min(xz_hist_data[xz_hist_data > 0]),
            np.min(cyl_hist_data[cyl_hist_data > 0]),
        ]
    ), max(
        np.nanmax(xy_hist_data), np.nanmax(xz_hist_data), np.nanmax(cyl_hist_data)
    )  # Ensure minimum is at least 1 for cmin

    # Plot histograms with the shared color scale
    xy_hist = xy_axis.pcolormesh(
        x_bins,
        y_bins,
        xy_hist_data.T,
        vmin=vmin,
        vmax=vmax,
        norm="log",
    )
    xz_hist = xz_axis.pcolormesh(
        x_bins,
        z_bins,
        xz_hist_data.T,
        vmin=vmin,
        vmax=vmax,
        norm="log",
    )
    cyl_hist = cyl_axis.pcolormesh(
        x_bins,
        cyl_bins,
        cyl_hist_data.T,
        vmin=vmin,
        vmax=vmax,
        norm="log",
    )

    xy_axis.set_xlabel(r"$X_{\rm MSM'}$ [$R_M$]")
    xy_axis.set_ylabel(r"$Y_{\rm MSM'}$ [$R_M$]")

    xz_axis.set_xlabel(r"$X_{\rm MSM'}$ [$R_M$]")
    xz_axis.set_ylabel(r"$Z_{\rm MSM'}$ [$R_M$]")

    for ax in axes[:-1]:
        plotting.Plot_Magnetospheric_Boundaries(ax, lw=2, zorder=5)
        ax.set_aspect("equal")

    plotting.Plot_Mercury(xy_axis, lw=2)
    plotting.Plot_Mercury(xz_axis, plane="xz", frame="MSM", lw=2)

    # Format cylindrical plot
    cyl_axis.set_xlabel(r"$X_{\text{MSM'}} \quad \left[ \text{R}_\text{M} \right]$")
    cyl_axis.set_ylabel(
        r"$\left( Y_{\text{MSM'}}^2 + Z_{\text{MSM'}}^2 \right)^{0.5} \quad \left[ \text{R}_\text{M} \right]$"
    )

    plotting.Plot_Circle(
        cyl_axis,
        (0, +utils.Constants.DIPOLE_OFFSET_RADII),
        1,
        shade_half=False,
        lw=2,
        ec="k",
        color="none",
    )
    plotting.Plot_Circle(
        cyl_axis,
        (0, -utils.Constants.DIPOLE_OFFSET_RADII),
        1,
        shade_half=False,
        lw=2,
        ec="k",
        color="none",
    )

    cyl_axis.set_aspect("equal")
    plotting.Plot_Magnetospheric_Boundaries(cyl_axis, lw=2, zorder=5)

    # Create a new axis above the subplots for the colorbar
    if i == 0:
        cbar_ax = fig.add_axes(
            (0.91, 0.54, 0.01, 0.33)
        )  # [left, bottom, width, height]
    else:
        cbar_ax = fig.add_axes(
            (0.91, 0.13, 0.01, 0.33)
        )  # [left, bottom, width, height]

    # Add colorbar
    cbar = fig.colorbar(xy_hist, cax=cbar_ax, orientation="vertical")

    label_x_offset = 0.065

    if i == 0:
        cbar.set_label(r"Bow Shock Crossing Rate [$\text{hour}^{-1}$]")
        fig.text(0.075, 0.84, "(a)", fontsize=14, transform=fig.transFigure)
        fig.text(0.375, 0.84, "(b)", fontsize=14, transform=fig.transFigure)
        fig.text(0.675, 0.84, "(c)", fontsize=14, transform=fig.transFigure)
    else:
        cbar.set_label(r"Magnetopause Crossing Rate [$\text{hour}^{-1}$]")
        fig.text(0.075, 0.43, "(d)", fontsize=14, transform=fig.transFigure)
        fig.text(0.375, 0.43, "(e)", fontsize=14, transform=fig.transFigure)
        fig.text(0.675, 0.43, "(f)", fontsize=14, transform=fig.transFigure)

    for ax in axes:
        ax.set_xlim(x_bins[0], x_bins[-1])

    xy_axis.set_ylim(y_bins[0], y_bins[-1])
    xz_axis.set_ylim(z_bins[0], z_bins[-1])
    cyl_axis.set_ylim(cyl_bins[0], cyl_bins[-1])


fig.subplots_adjust(left=0.07, top=0.9, bottom=0.1, wspace=0.3, hspace=0.05)
plt.show()

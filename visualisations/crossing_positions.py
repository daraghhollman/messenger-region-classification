import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy import plotting, trajectory, utils

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

bow_shock_locations = (
    trajectory.Get_Position(
        "MESSENGER", bow_shock_crossings["Time"], frame="MSM", aberrate="average"
    )
    / utils.Constants.MERCURY_RADIUS_KM
)
magnetopause_locations = (
    trajectory.Get_Position(
        "MESSENGER", magnetopause_crossings["Time"], frame="MSM", aberrate="average"
    )
    / utils.Constants.MERCURY_RADIUS_KM
)


fig, axes = plt.subplots(2, 3, figsize=(8, 6))

bow_shock_axes = axes[0]
magnetopause_axes = axes[1]

for i, (axes, positions) in enumerate(
    zip(
        [bow_shock_axes, magnetopause_axes],
        [bow_shock_locations, magnetopause_locations],
    )
):

    xy_axis, xz_axis, cyl_axis = axes

    bin_size = 0.5
    x_bins = np.arange(-5, 5 + bin_size, bin_size)
    y_bins = np.arange(-5, 5 + bin_size, bin_size)
    z_bins = np.arange(-8, 2 + bin_size, bin_size)
    cyl_bins = np.arange(0, 10 + bin_size, bin_size)

    xy_hist_data, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=[x_bins, y_bins]
    )
    xz_hist_data, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 2], bins=[x_bins, z_bins]
    )
    cyl_hist_data, _, _ = np.histogram2d(
        positions[:, 0],
        np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2),
        bins=[x_bins, cyl_bins],
    )

    # Determine the global vmin and vmax
    vmin, vmax = 1, max(
        xy_hist_data.max(), xz_hist_data.max(), cyl_hist_data.max()
    )  # Ensure minimum is at least 1 for cmin

    # Plot histograms with the shared color scale
    xy_hist = xy_axis.hist2d(
        positions[:, 0],
        positions[:, 1],
        bins=[x_bins, y_bins],
        cmin=1,
        vmin=vmin,
        vmax=vmax,
    )
    xz_hist = xz_axis.hist2d(
        positions[:, 0],
        positions[:, 2],
        bins=[x_bins, z_bins],
        cmin=1,
        vmin=vmin,
        vmax=vmax,
    )
    cyl_hist = cyl_axis.hist2d(
        positions[:, 0],
        np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2),
        bins=[x_bins, cyl_bins],
        cmin=1,
        vmin=vmin,
        vmax=vmax,
    )

    xy_axis.set_xlabel("$X_{MSM'}$ [$R_m$]")
    xy_axis.set_ylabel("$Y_{MSM'}$ [$R_m$]")

    xz_axis.set_xlabel("$X_{MSM'}$ [$R_m$]")
    xz_axis.set_ylabel("$Z_{MSM'}$ [$R_m$]")

    for ax in axes[:-1]:
        plotting.Plot_Magnetospheric_Boundaries(ax, lw=2, zorder=5)
        ax.set_aspect("equal")

    plotting.Plot_Mercury(xy_axis, lw=2)
    plotting.Plot_Mercury(xz_axis, plane="xz", frame="MSM", lw=2)

    # Format cylindrical plot
    cyl_axis.set_xlabel(
        r"$\text{X}_{\text{MSM'}} \quad \left[ \text{R}_\text{M} \right]$"
    )
    cyl_axis.set_ylabel(
        r"$\left( \text{Y}_{\text{MSM'}}^2 + \text{Z}_{\text{MSM'}}^2 \right)^{0.5} \quad \left[ \text{R}_\text{M} \right]$"
    )

    plotting.Plot_Circle(cyl_axis, (0, +utils.Constants.DIPOLE_OFFSET_RADII), 1, shade_half=False, lw=2, ec="k", color="none")
    plotting.Plot_Circle(cyl_axis, (0, -utils.Constants.DIPOLE_OFFSET_RADII), 1, shade_half=False, lw=2, ec="k", color="none")

    cyl_axis.set_aspect("equal")
    plotting.Plot_Magnetospheric_Boundaries(cyl_axis, lw=2, zorder=5)

    # Create a new axis above the subplots for the colorbar
    if i == 0:
        cbar_ax = fig.add_axes((0.3, 0.95, 0.4, 0.02))  # [left, bottom, width, height]
    else:
        cbar_ax = fig.add_axes((0.3, 0.45, 0.4, 0.02))  # [left, bottom, width, height]

    # Add colorbar
    cbar = fig.colorbar(xy_hist[3], cax=cbar_ax, orientation="horizontal")

    if i == 0:
        cbar.set_label("Number of Bow Shock Crossings")
        fig.text(0.05, 0.95, "(a)", fontsize=14, transform=fig.transFigure)
    else:
        cbar.set_label("Number of Magnetopause Crossings")
        fig.text(0.05, 0.45, "(b)", fontsize=14, transform=fig.transFigure)


fig.subplots_adjust(top=0.9, bottom=0.05, wspace=0.4, hspace=0.5)
plt.savefig("/home/daraghhollman/Main/Work/papers/boundaries/figures/new_crossing_spatial_spread.pdf", format="pdf")

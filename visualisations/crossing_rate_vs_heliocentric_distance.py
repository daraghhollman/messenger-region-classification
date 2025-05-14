import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy import boundaries, mag, trajectory, utils

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

crossings["Heliocentric Distance"] = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(crossings["Time"])
)

full_mission["Heliocentric Distance"] = utils.Constants.KM_TO_AU(
    trajectory.get_heliocentric_distances_parallel(full_mission["date"], processes=20)
)

bin_size = 0.5
x_bins = np.arange(-5, 5 + bin_size, bin_size)
y_bins = np.arange(-5, 5 + bin_size, bin_size)
z_bins = np.arange(-8, 2 + bin_size, bin_size)
cyl_bins = np.arange(0, 10 + bin_size, bin_size)

# Limit by heliocentric distance bin
heliocentric_distance_bin_edges = np.linspace(0.3, 0.47, 17)

bow_shock_crossing_rates = {
    "Mean": [],
    "Median": [],
}
magnetopause_crossing_rates = {
    "Mean": [],
    "Median": [],
}

for bin_start, bin_end in zip(
    heliocentric_distance_bin_edges, heliocentric_distance_bin_edges[1:]
):

    filtered_mission = full_mission.loc[
        full_mission["Heliocentric Distance"].between(bin_start, bin_end)
    ]

    positions = [
        filtered_mission["X MSM' (radii)"],
        filtered_mission["Y MSM' (radii)"],
        filtered_mission["Z MSM' (radii)"],
    ]

    filtered_crossings = crossings.loc[
        crossings["Heliocentric Distance"].between(bin_start, bin_end)
    ]

    bow_shock_crossings = filtered_crossings.loc[
        filtered_crossings["Transition"].str.contains("BS")
    ].copy()
    magnetopause_crossings = filtered_crossings.loc[
        filtered_crossings["Transition"].str.contains("MP")
    ].copy()

    # Get residence histograms. These are the frequency of data points. We have
    # loaded 1 second average data.
    residence_xy, _, _ = np.histogram2d(
        positions[0], positions[1], bins=[x_bins, y_bins]
    )
    residence_xz, _, _ = np.histogram2d(
        positions[0], positions[2], bins=[x_bins, z_bins]
    )
    residence_cyl, _, _ = np.histogram2d(
        positions[0],
        np.sqrt(positions[1] ** 2 + positions[2] ** 2),
        bins=[x_bins, cyl_bins],
    )

    # Loop for magnetopause and bow shock
    for i, c in enumerate([bow_shock_crossings, magnetopause_crossings]):

        xy_hist_data, _, _ = np.histogram2d(
            c["X MSM' (radii)"],
            c["Y MSM' (radii)"],
            bins=[x_bins, y_bins],
        )
        xz_hist_data, _, _ = np.histogram2d(
            c["X MSM' (radii)"],
            c["Z MSM' (radii)"],
            bins=[x_bins, z_bins],
        )
        cyl_hist_data, _, _ = np.histogram2d(
            c["X MSM' (radii)"],
            np.sqrt(c["Y MSM' (radii)"] ** 2 + c["Z MSM' (radii)"] ** 2),
            bins=[x_bins, cyl_bins],
        )

        # Normalise
        # Yielding crossings per second

        xy_hist_data = np.where(residence_xy != 0, xy_hist_data / residence_xy, np.nan)
        xz_hist_data = np.where(residence_xz != 0, xz_hist_data / residence_xz, np.nan)
        cyl_hist_data = np.where(
            residence_cyl != 0, cyl_hist_data / residence_cyl, np.nan
        )

        # Limit to non-zero, non-nan data
        valid_data = cyl_hist_data[~np.isnan(cyl_hist_data) & (cyl_hist_data > 0)]

        median_crossing_rate = np.median(valid_data)
        mean_crossing_rate = np.mean(valid_data)

        if i == 0:
            bow_shock_crossing_rates["Median"].append(median_crossing_rate)
            bow_shock_crossing_rates["Mean"].append(mean_crossing_rate)
        else:
            magnetopause_crossing_rates["Median"].append(median_crossing_rate)
            magnetopause_crossing_rates["Mean"].append(mean_crossing_rate)

bin_centres = (
    heliocentric_distance_bin_edges[:-1] + heliocentric_distance_bin_edges[1:]
) / 2

fig, ax = plt.subplots()

ax.scatter(
    bin_centres,
    bow_shock_crossing_rates["Mean"],
    color="black",
    label="BS Mean Crossing Rate",
)
ax.scatter(
    bin_centres,
    bow_shock_crossing_rates["Median"],
    color="grey",
    label="BS Median Crossing Rate",
)

ax.scatter(
    bin_centres,
    magnetopause_crossing_rates["Mean"],
    color=wong_colours["blue"],
    label="MP Mean Crossing Rate",
)
ax.scatter(
    bin_centres,
    magnetopause_crossing_rates["Median"],
    color=wong_colours["light blue"],
    label="MP Median Crossing Rate",
)

ax.set_xlabel("Heliocentric Distance [AU]")
ax.set_ylabel("Residence-normalised Average Crossing Rate [crossings per second]")

ax.legend()

# plt.show()


# We want to multiply these by the Philpott interval durations to see if they
# match the counts in crossings
# Load Philpott intervals
philpott_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False, backend="Philpott"
)

philpott_intervals["Duration"] = (
    philpott_intervals["End Time"] - philpott_intervals["Start Time"]
).dt.total_seconds()

philpott_intervals["Mid Time"] = (
    philpott_intervals["Start Time"]
    + (philpott_intervals["End Time"] - philpott_intervals["Start Time"]) / 2
)
philpott_intervals["Heliocentric Distance (AU)"] = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(philpott_intervals["Mid Time"])
)

bs_intervals = philpott_intervals.loc[philpott_intervals["Type"].str.contains("BS")]
mp_intervals = philpott_intervals.loc[philpott_intervals["Type"].str.contains("MP")]

for i, intervals in enumerate([bs_intervals, mp_intervals]):

    intervals["Distance Bin"] = pd.cut(
        intervals["Heliocentric Distance (AU)"],
        bins=heliocentric_distance_bin_edges,
    )

    grouped_durations = [
        group["Duration"].values for _, group in intervals.groupby("Distance Bin")
    ]

    mean_duration_per_bin = np.array(
        [np.mean(durations) for durations in grouped_durations]
    )

    if i == 0:
        bs_mean_durations = mean_duration_per_bin
        bs_intervals_per_bin = intervals.groupby("Distance Bin").size()

    else:
        mp_mean_durations = mean_duration_per_bin
        mp_intervals_per_bin = intervals.groupby("Distance Bin").size()


expected_bs_crossings = np.multiply(bs_mean_durations, bow_shock_crossing_rates["Mean"])
expected_mp_crossings = np.multiply(
    mp_mean_durations, magnetopause_crossing_rates["Mean"]
)

expected_bs_crossings *= bs_intervals_per_bin
expected_mp_crossings *= mp_intervals_per_bin

print(f"Total expected BS crossings: {sum(expected_bs_crossings)}")
print(f"Total expected MP crossings: {sum(expected_mp_crossings)}")

fig, ax = plt.subplots()

ax.plot(bin_centres, expected_bs_crossings, color="black", label="Bow Shock")
ax.plot(
    bin_centres, expected_mp_crossings, color=wong_colours["blue"], label="Magnetopause"
)

ax.set_xlabel("Heliocentric Distance [AU]")

ax.set_ylabel(
    "Expected Number of Crossings\n( Mean Crossing Rate × Mean Interval Duration × Number of Intervals ) per bin"
)

ax.legend()

plt.show()

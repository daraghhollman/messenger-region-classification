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

# To quickly find the position of each crossing in MSM' coordinates, we can
# load the full mission at 1 second resolution from file, and cross-reference.
full_mission = mag.Load_Mission("/home/daraghhollman/Main/data/mercury/messenger_mag")
crossings = pd.merge_asof(
    # Does this using the nearest element
    # Its so fast omg
    crossings,
    full_mission,
    left_on="Time",
    right_on="date",
    direction="nearest",
)

# We want to filter by heliocentric distance, and so must calculate this for
# each crossing
crossings["Heliocentric Distance"] = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(crossings["Time"])
)


# We want to calculate heliocentric distance for all rows of the full mission,
# however, there are far too many calculations at a one second cadence. Hence,
# we need to bin the values at some frequency, and determine the Heliocentric
# distance for that bin alone.
# This is some fancy code to do that in a reasonable amount of time
def Get_Mission_Heliocentric_Distance(mission, frequency):
    bin_frequency = frequency
    bin_start = mission["date"].iloc[0].floor("D")
    bin_end = mission["date"].iloc[-1].ceil("D")
    bin_edges = pd.date_range(bin_start, bin_end, freq=bin_frequency)

    bin_indices = np.searchsorted(bin_edges, mission["date"], side="right") - 1
    mission["Bin Index"] = bin_indices

    bin_centres = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

    bin_heliocentric_distances = utils.Constants.KM_TO_AU(
        trajectory.Get_Heliocentric_Distance(bin_centres)
    )

    mission["Bin Index"] = np.clip(bin_indices, 0, len(bin_heliocentric_distances) - 1)

    mission["Heliocentric Distance"] = bin_heliocentric_distances[
        mission["Bin Index"].values
    ]


# Calculate mission heliocentric distance, binned weekly
Get_Mission_Heliocentric_Distance(full_mission, "1W")

# Define spatial bins
bin_size = 0.5
x_bins = np.arange(-5, 5 + bin_size, bin_size)
y_bins = np.arange(-5, 5 + bin_size, bin_size)
z_bins = np.arange(-8, 2 + bin_size, bin_size)
cyl_bins = np.arange(0, 10 + bin_size, bin_size)

# Define heliocentric distance bins
heliocentric_distance_bin_size = 0.01
heliocentric_distance_bin_edges = np.arange(
    0.3, 0.47 + heliocentric_distance_bin_size, heliocentric_distance_bin_size
)
heliocentric_distance_bin_centres = (
    heliocentric_distance_bin_edges[:-1] + heliocentric_distance_bin_edges[1:]
) / 2

bow_shock_crossing_rates = []

# Loop through each heliocentric distance bin
for bin_start, bin_end in zip(
    heliocentric_distance_bin_edges, heliocentric_distance_bin_edges[1:]
):

    # Filter for only the crossings within this bin
    filtered_crossings = crossings.loc[
        crossings["Heliocentric Distance"].between(bin_start, bin_end)
    ]

    # Split by BS or MP
    bow_shock_crossings = filtered_crossings.loc[
        filtered_crossings["Transition"].str.contains("BS")
    ].copy()
    magnetopause_crossings = filtered_crossings.loc[
        filtered_crossings["Transition"].str.contains("MP")
    ].copy()

    filtered_mission = full_mission.loc[
        full_mission["Heliocentric Distance"].between(bin_start, bin_end)
    ]
    positions = [
        filtered_mission["X MSM' (radii)"],
        filtered_mission["Y MSM' (radii)"],
        filtered_mission["Z MSM' (radii)"],
    ]

    # Find the spatial spread of bow shock crossings in the cylindrical
    # coordinate system for this heliocentric bin
    bow_shock_spatial_counts, _, _ = np.histogram2d(
        bow_shock_crossings["X MSM' (radii)"],
        np.sqrt(
            bow_shock_crossings["Y MSM' (radii)"] ** 2
            + bow_shock_crossings["Z MSM' (radii)"] ** 2
        ),
        bins=[x_bins.tolist(), cyl_bins.tolist()],
    )

    """
    # Little snippet to view the bow shock bin counts
    plt.pcolormesh(x_bins, cyl_bins, bow_shock_spatial_counts.T)
    plt.colorbar()
    plt.show()
    continue
    """

    # We want to know how much time the spacecraft spent in each spatial bin
    # for this heliocentric distance bin
    # Each data point is a second in time
    residence, _, _ = np.histogram2d(
        positions[0],
        np.sqrt(positions[1] ** 2 + positions[2] ** 2),
        bins=[x_bins.tolist(), cyl_bins.tolist()],
    )

    """
    # Little snippet to view the residence
    plt.pcolormesh(x_bins, cyl_bins, residence.T)
    plt.colorbar()
    plt.show()
    continue
    """

    # We then divide the spatial crossing density by the dwell time in each bin
    # We ignore where residence == 0 as these are invalid
    # This gives crossings per second for each spatial bin for each
    # heliocentric distance bin
    with np.errstate(divide="ignore", invalid="ignore"):
        # We can safely disable the divide by zero warning here as we use a
        # where statement. The where still does the full calculation, which is
        # why the warning is there, but the values areplaced with nan
        bow_shock_crossing_rate = np.where(
            residence != 0, bow_shock_spatial_counts / residence, np.nan
        )

    """
    # Little snippet to view the crossing rate
    plt.pcolormesh(x_bins, cyl_bins, bow_shock_crossing_rate.T)
    plt.colorbar()
    plt.show()
    continue
    """

    # Total crossing rate is the sum of the crossing rate in all spatial bins
    total_bow_shock_crossing_rate = np.nansum(bow_shock_crossing_rate)
    bow_shock_crossing_rates.append(total_bow_shock_crossing_rate)

    # We want to multiply this by the total Philpott interval duration within that bin


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

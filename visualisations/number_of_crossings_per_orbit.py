"""
Script to investigate how the number of new crossings per orbit changes in time
"""

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from hermpy import trajectory, utils
from hermpy.plotting import wong_colours

# Load new crossings
crossings = pd.read_csv(
    # "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings.csv"
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings_with_adjusted_counting.csv"
)
crossings["Time"] = pd.to_datetime(crossings["Time"])

# Get the orbit of each crossing
crossings["Orbit Number"] = trajectory.Get_Orbit_Number(crossings["Time"])

crossings_per_orbit = crossings.groupby(["Orbit Number"]).size()
orbit_mean_times = crossings.groupby(["Orbit Number"])["Time"].mean()

bs_crossings_per_orbit = (
    crossings.loc[crossings["Transition"].str.contains("BS")]
    .groupby(["Orbit Number"])
    .size()
)
bs_orbit_mean_times = (
    crossings.loc[crossings["Transition"].str.contains("BS")]
    .groupby(["Orbit Number"])["Time"]
    .mean()
)

mp_crossings_per_orbit = (
    crossings.loc[crossings["Transition"].str.contains("MP")]
    .groupby(["Orbit Number"])
    .size()
)
mp_orbit_mean_times = (
    crossings.loc[crossings["Transition"].str.contains("MP")]
    .groupby(["Orbit Number"])["Time"]
    .mean()
)

# As we define orbit number based on the crossing intervals, this output is techincally invalid
# for the first orbit. We must remove the first row.
orbit_mean_times = orbit_mean_times.iloc[1:]
bs_orbit_mean_times = bs_orbit_mean_times.iloc[1:]
mp_orbit_mean_times = mp_orbit_mean_times.iloc[1:]

crossings_per_orbit = crossings_per_orbit.iloc[1:]
bs_crossings_per_orbit = bs_crossings_per_orbit.iloc[1:]
mp_crossings_per_orbit = mp_crossings_per_orbit.iloc[1:]

fig, ax = plt.subplots()
heliocentric_distance_ax = ax.twinx()
heliocentric_distance_ax.margins(0)
heliocentric_distance_ax.yaxis.label.set_color(wong_colours["red"])
heliocentric_distance_ax.tick_params(axis="y", colors=wong_colours["red"])

ax.spines["right"].set_color(wong_colours["red"])
ax.zorder = 5
ax.patch.set_visible(False)  # Remove axis background

start_date = crossings["Time"].iloc[0]
end_date = crossings["Time"].iloc[-1]

heliocentric_distance = utils.Constants.KM_TO_AU(
    trajectory.Get_Heliocentric_Distance(orbit_mean_times)
)

heliocentric_distance_ax.plot(
    orbit_mean_times,
    heliocentric_distance,
    color=wong_colours["red"],
    ls="dashed",
    label="Mercury Heliocentric Distance",
)
heliocentric_distance_ax.fill_between(
    orbit_mean_times, 0.3, heliocentric_distance, color=wong_colours["red"], alpha=0.1
)

heliocentric_distance_ax.set_ylabel("Heliocentric Distance [AU]")

# ax.plot(orbit_mean_times, crossings_per_orbit, color="lightgrey", label="All Crossings")
ax.plot(bs_orbit_mean_times, bs_crossings_per_orbit, color="black", label="Bow Shock")
ax.plot(
    mp_orbit_mean_times,
    mp_crossings_per_orbit,
    color=wong_colours["light blue"],
    label="Magnetopause",
)

ax.set_xlabel("Time")
ax.set_ylabel("Crossings per Orbit")
ax.legend()
ax.margins(0)

# ax.set_yscale("log")

plt.show()

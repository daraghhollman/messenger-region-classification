"""
Script to plot any section of time from MESSENGER MAG and show new crossings
"""

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from hermpy import mag, plotting, utils
from hermpy.plotting import wong_colours

# Define the plot
start_time = "2013-02-01 00:00:00"
end_time = "2013-02-10 00:00:00"

# Convert to datetime obj
start_time = dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
end_time = dt.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

# Load MESSENGER MAG mission
messenger_data = mag.Load_Mission(utils.User.DATA_DIRECTORIES["FULL MISSION"])

# Strip this down to between the times
messenger_data = messenger_data.loc[
    messenger_data["date"].between(start_time, end_time)
]

# Load new crossing locations
hollman_crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/post-processing/2_hidded_crossings_included.csv",
    parse_dates=["Times"],
)

# Find only crossings within the times
relevant_crossings = hollman_crossings.loc[
    hollman_crossings["Times"].between(start_time, end_time)
]

# Create figure
fig, ax = plt.subplots()

ax.plot(messenger_data["date"], messenger_data["|B|"], color="black", label="|B|")
ax.plot(
    messenger_data["date"], messenger_data["Bx"], color=wong_colours["red"], label="Bx"
)
ax.plot(
    messenger_data["date"],
    messenger_data["By"],
    color=wong_colours["green"],
    label="By",
)
ax.plot(
    messenger_data["date"], messenger_data["Bz"], color=wong_colours["blue"], label="Bz"
)

map_crossing_to_region = {
    # Match the region before to the crossing name
    "BS_IN": wong_colours["yellow"],
    "BS_OUT": wong_colours["orange"],
    "MP_IN": wong_colours["orange"],
    "MP_OUT": wong_colours["light blue"],
}

# Add crossings
for i, crossing in relevant_crossings.iterrows():

    previous_crossing = hollman_crossings.loc[i - 1]

    ax.axvline(crossing["Times"], color=wong_colours["pink"], zorder=5)

    ax.axvspan(
        previous_crossing["Times"],
        crossing["Times"],
        color=map_crossing_to_region[crossing["Label"]],
        alpha=0.8,
        zorder=-1,
    )


ax.legend()

plt.show()

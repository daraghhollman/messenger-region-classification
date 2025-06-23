"""
Script to visualise if the first post-processing step functions correctly
"""

import datetime as dt
import random

import matplotlib.patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from hermpy import boundaries, mag, plotting, utils
from hermpy.plotting import wong_colours

# Load Philpott crossing intervals and define crossing groups
print("Loading crossings intervals")
crossing_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=True
)

print("Grouping crossing intervals")
crossing_groups = []
crossing_index = 0
while crossing_index < len(crossing_intervals) - 1:

    current_crossing = crossing_intervals.loc[crossing_index]
    next_crossing = crossing_intervals.loc[crossing_index + 1]

    assert isinstance(current_crossing, pd.Series)

    if current_crossing["Type"] == "BS_IN":
        # We expect a magnetopause in crossing next
        match next_crossing["Type"]:
            case "MP_IN":
                # This is as normal, we can add to our list of pairs
                crossing_groups.append([current_crossing, next_crossing])

                # We don't want to consider the next crossing as we have already
                # saved it, so we add an extra to the crossing index.
                crossing_index += 1

            case label if label in ["MP_OUT", "BS_IN", "BS_OUT", "DATA_GAP"]:
                # This is abnormal, we just want to look around the current crossing
                crossing_groups.append([current_crossing])

    elif current_crossing["Type"] == "MP_OUT":
        # We expect a bow shock in crossing next
        match next_crossing["Type"]:
            case "BS_OUT":
                # This is as normal, we can add to our list of pairs
                crossing_groups.append([current_crossing, next_crossing])

                # We don't want to consider the next crossing as we have already
                # saved it, so we add an extra to the crossing index.
                crossing_index += 1

            case label if label in ["MP_IN", "MP_OUT", "BS_IN", "DATA_GAP"]:
                # This is abnormal, we just want to look around the current crossing
                crossing_groups.append([current_crossing])

    else:
        # Otherwise, for some reason the previous part of the crossing pair
        # didn't exist. We save this crossing on its own.
        if current_crossing["Type"] != "DATA_GAP":
            crossing_groups.append([current_crossing])

    crossing_index += 1


# Load post-processed regions
model_regions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/post-processing/1_bookend_regions_processed.csv",
    parse_dates=["Start Time", "End Time"],
)

# Randomly sort the crossing groups and loop through them to look at
random.shuffle(crossing_groups)

for crossing_group in crossing_groups:

    # Load data around the interval
    interval_buffer = dt.timedelta(minutes=10)

    if len(crossing_group) == 1:
        start = crossing_group[0]["Start Time"] - interval_buffer
        end = crossing_group[0]["End Time"] + interval_buffer

    else:
        start = crossing_group[0]["Start Time"] - interval_buffer
        end = crossing_group[1]["End Time"] + interval_buffer

    print("Loading data")
    messenger_data = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG"], start, end
    )

    # Find regions within this data
    regions_in_data = model_regions.loc[
        model_regions["Start Time"].between(start, end)
    ].reset_index(drop=True)

    # Create a figure and plot the mag data
    print("Creating plot")
    fig, axes = plt.subplots(2, 1, sharex=True)
    (magnitude_axis, components_axis) = axes

    # Plot the magnetic field components
    for component, component_label, colour in zip(
        ["Bx", "By", "Bz"], ["$B_x$", "$B_y$", "$B_z$"], ["red", "green", "blue"]
    ):
        components_axis.plot(
            messenger_data["date"],
            messenger_data[component],
            color=wong_colours[colour],
            label=component_label,
        )

    magnitude_axis.plot(
        messenger_data["date"],
        messenger_data["|B|"],
        color=wong_colours["black"],
        label="$|B|$",
    )

    magnitude_axis.set_ylabel("|B| [nT]")

    components_axis.set_ylabel("Magnetic Field Strength [nT]")
    components_axis.axhline(0, color="black", ls="dotted", lw=2)
    components_legend = components_axis.legend()
    components_legend = components_axis.legend()

    for ax in axes:
        ax.margins(0)

    plotting.Add_Tick_Ephemeris(components_axis)

    # Add boundary crossing intervals
    # We only need start time within the data as crossing groups never spans
    # part of an interval only
    intervals_within_data = crossing_intervals.loc[
        crossing_intervals["Start Time"].between(start, end)
    ]

    for _, crossing_interval in intervals_within_data.iterrows():
        for ax in axes:
            ax.axvspan(
                crossing_interval["Start Time"],
                crossing_interval["End Time"],
                fill=False,
                linewidth=0,
                hatch="/",
                zorder=5,
            )

    # Add region shading
    colour_table = {
        "Solar Wind": wong_colours["yellow"],
        "Magnetosheath": wong_colours["orange"],
        "Magnetosphere": wong_colours["blue"],
        "Unknown": "lightgrey",
    }
    for _, region in regions_in_data.iterrows():

        for ax in axes:
            ax.axvspan(
                region["Start Time"],
                region["End Time"],
                color=colour_table[region["Label"]],
                zorder=-1,
            )

    for ax in axes:
        # Format ticks
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        ax.tick_params(
            "x", which="major", direction="inout", length=20, width=1.5, zorder=5
        )
        ax.tick_params(
            "x", which="minor", direction="inout", length=10, width=1.5, zorder=5
        )

        ax.tick_params("y", which="major", direction="out", length=10)
        ax.tick_params("y", which="minor", direction="out", length=5)

    print("Displaying plot")
    plt.show()

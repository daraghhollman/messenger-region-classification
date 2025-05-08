"""
Script to visualise the application of any model to a given crossing group or section of time
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
from pandas._libs import interval

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


# Load probabilities
model_output = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/output.csv"
)
model_output["Time"] = pd.to_datetime(model_output["Time"], format="ISO8601")

# Load the new crossing list
new_crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings.csv"
)
new_crossings["Time"] = pd.to_datetime(new_crossings["Time"])

# Randomly sort the crossing groups and loop through them to look at
# random.shuffle(crossing_groups)

i = 0
for crossing_group in crossing_groups:

    print(i)

    # Load data around the interval
    interval_buffer = dt.timedelta(minutes=10)

    if isinstance(crossing_group, pd.Series):
        start = crossing_group["Start Time"] - interval_buffer
        end = crossing_group["End Time"] + interval_buffer

    else:
        start = crossing_group[0]["Start Time"] - interval_buffer
        end = crossing_group[1]["End Time"] + interval_buffer

    print("Loading data")
    messenger_data = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG"], start, end
    )

    # Get model_ouput between these times
    probabilities = model_output.loc[model_output["Time"].between(start, end)]

    # Search the model output for new crossings in this interval
    crossings_in_data = new_crossings.loc[
        new_crossings["Time"].between(start, end)
    ].reset_index(drop=True)

    # Create a figure and plot the mag data
    print("Creating plot")
    fig, axes = plt.subplots(3, 1, sharex=True)
    (magnitude_axis, components_axis, probability_axis) = axes

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

    probability_axis.plot(
        probabilities["Time"],
        probabilities["P(SW)"],
        color=wong_colours["yellow"],
        path_effects=[  # Add a black outline to the line
            matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
            matplotlib.patheffects.Normal(),
        ],
        label="P(SW)",
    )
    probability_axis.plot(
        probabilities["Time"],
        probabilities["P(MSh)"],
        color=wong_colours["orange"],
        path_effects=[  # Add a black outline to the line
            matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
            matplotlib.patheffects.Normal(),
        ],
        label="P(MSh)",
    )
    probability_axis.plot(
        probabilities["Time"],
        probabilities["P(MSp)"],
        color=wong_colours["blue"],
        path_effects=[  # Add a black outline to the line
            matplotlib.patheffects.Stroke(linewidth=2, foreground="k"),
            matplotlib.patheffects.Normal(),
        ],
        label="P(MSp)",
    )
    probability_axis.legend()
    probability_axis.set_ylim(0, 1)
    probability_axis.set_ylabel("Class Probability")
    plotting.Add_Tick_Ephemeris(probability_axis)

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

    # Plot new crossings

    # Text-box formatting
    text_box_formatting = dict(
        boxstyle="square", facecolor="white", edgecolor="black", pad=0, alpha=1
    )

    crossing_labels = []
    for index, c in crossings_in_data.iterrows():

        for ax in axes:
            ax.axvline(c["Time"], color="black", ls="dashed")

        assert isinstance(index, int)

        label_y = np.linspace(1.05, 1.3, 4)[index % 4]
        crossing_label = magnitude_axis.text(
            c["Time"],
            label_y,
            c["Transition"] if "UKN" not in c["Transition"] else "UKN",
            va="bottom",
            ha="center",
            fontweight="bold",
            fontsize="small",
            transform=magnitude_axis.get_xaxis_transform(),
            bbox=text_box_formatting,
        )
        crossing_labels.append(crossing_label)

        # Add a line between label and axis
        line = magnitude_axis.axvline(
            c["Time"],
            1,
            label_y,
            color="black",
            ls="dashed",
        )
        line.set_clip_on(False)

        if index == 0:
            # Shade the region before the first crossing
            match c["Transition"]:

                case "BS_OUT" | "UKN (MSh -> UKN)" | "MP_IN":
                    # Region was magnetosheath
                    shade = wong_colours["orange"]

                case "BS_IN" | "UKN (SW -> UKN)":
                    # Region was solar wind
                    shade = wong_colours["yellow"]

                case "MP_OUT" | "UKN (MSp -> UKN)":
                    # Region was magnetosphere
                    shade = wong_colours["blue"]

                case _:
                    shade = "white"

            for ax in axes:
                ax.axvspan(start, c["Time"], color=shade, alpha=0.3)

            if len(crossings_in_data) == 1:
                # This is the only crossing
                # So we need to shade the next region too
                # Shade between the current crossing and the next
                match c["Transition"]:

                    case "BS_OUT" | "UKN (UKN -> SW)":
                        # Region is solar wind
                        shade = wong_colours["yellow"]

                    case "BS_IN" | "MP_OUT" | "UKN (UKN -> MSh)":
                        # Region is magnetosheath
                        shade = wong_colours["orange"]

                    case "MP_IN" | "UKN (UKN -> MSp)":
                        # Region is magnetosphere
                        shade = wong_colours["blue"]

                    case _:
                        shade = "white"

                for ax in axes:
                    ax.axvspan(c["Time"], end, color=shade, alpha=0.3)

        if index < len(crossings_in_data) - 1:

            # Shade between the current crossing and the next
            match c["Transition"]:

                case "BS_OUT" | "UKN (UKN -> SW)":
                    # Region is solar wind
                    shade = wong_colours["yellow"]

                case "BS_IN" | "MP_OUT" | "UKN (UKN -> MSh)":
                    # Region is magnetosheath
                    shade = wong_colours["orange"]

                case "MP_IN" | "UKN (UKN -> MSp)":
                    # Region is magnetosphere
                    shade = wong_colours["blue"]

                case _:
                    shade = "lightgrey"

            for ax in axes:
                ax.axvspan(
                    c["Time"],
                    crossings_in_data.loc[index + 1]["Time"],
                    color=shade,
                    alpha=0.3,
                )

        elif index == len(crossings_in_data) - 1:

            # Shade between the current crossing and the next
            match c["Transition"]:

                case "BS_OUT" | "UKN (UKN -> SW)":
                    # Region is solar wind
                    shade = wong_colours["yellow"]

                case "BS_IN" | "MP_OUT" | "UKN (UKN -> MSh)":
                    # Region is magnetosheath
                    shade = wong_colours["orange"]

                case "MP_IN" | "UKN (UKN -> MSp)":
                    # Region is magnetosphere
                    shade = wong_colours["blue"]

                case _:
                    shade = "lightgrey"

            for ax in axes:
                ax.axvspan(c["Time"], end, color=shade, alpha=0.3)

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

    i += 1

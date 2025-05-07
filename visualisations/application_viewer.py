"""
Script to visualise the application of any model to a given crossing group or section of time
"""

import datetime as dt

import matplotlib.pyplot as plt
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
random_crossing_groups = crossing_intervals.sample(frac=1)

for i, crossing_group in random_crossing_groups.iterrows():

    # Load data around the interval
    interval_buffer = dt.timedelta(minutes=10)

    if isinstance(crossing_group, pd.Series):
        start = crossing_group["Start Time"] - interval_buffer
        end = crossing_group["End Time"] + interval_buffer

    else:
        print(type(crossing_group))
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
    components_axis.axhline(0, color="grey", ls="dotted", lw=2)
    components_legend = components_axis.legend()
    components_legend = components_axis.legend()

    for ax in axes:
        ax.margins(0)

    probability_axis.plot(
        probabilities["Time"],
        probabilities["P(SW)"],
        color=wong_colours["yellow"],
        lw=3,
        label="P(SW)",
    )
    probability_axis.plot(
        probabilities["Time"],
        probabilities["P(MSh)"],
        color=wong_colours["orange"],
        lw=3,
        label="P(MSh)",
    )
    probability_axis.plot(
        probabilities["Time"],
        probabilities["P(MSp)"],
        color=wong_colours["blue"],
        lw=3,
        label="P(MSp)",
    )
    probability_axis.legend()
    probability_axis.set_ylim(0, 1)
    plotting.Add_Tick_Ephemeris(probability_axis)

    # Add boundary crossing intervals
    """
    boundaries.Plot_Crossing_Intervals(
        ax,
        interval["Start Time"] - interval_buffer,
        interval["End Time"] + interval_buffer,
        crossing_intervals,
        lw=3,
        color="black",
    )
    """

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

    print("Displaying plot")
    plt.show()

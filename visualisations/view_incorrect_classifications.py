"""
Investigate crossing intervals where we classify the third i.e. the incorrect
region for that interval. i.e. We classify solar wind for a magnetopause
crossing interval
"""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hermpy import boundaries, mag, plotting, trajectory, utils
from hermpy.plotting import wong_colours

# We want to look randomly at crossing intervals from the Philpott list,
# and see where the new model placed crossings.
crossing_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False
)

# In this case, we want to specifically look at certain indices
specific_indices = [
    66,
    74,
    3206,  #
    3208,  # Consecutive entries might be a sign of something physically interesting
    3209,  #
    3406,
    3967,
    4844,
    4972,  # Again here too
    4973,  #
    5124,
    6084,
    10173,
    11124,
    14527,
]
crossing_intervals = crossing_intervals.loc[specific_indices]

# Load probabilities
model_output = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/output.csv"
)
model_output["Time"] = pd.to_datetime(model_output["Time"], format="ISO8601")

# Load the new crossing list
hollman_crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings.csv"
)
hollman_crossings["Time"] = pd.to_datetime(hollman_crossings["Time"])

# Load Pump crossings
pump_crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/pump_bowshock_bdry_detection_v2.csv"
)
pump_crossings["datetime"] = pd.to_datetime(pump_crossings["datetime"])
pump_crossings["Time"] = pump_crossings["datetime"]

for i, interval in crossing_intervals.iterrows():

    # Load data around the interval
    interval_buffer = dt.timedelta(minutes=10)

    start = interval["Start Time"] - interval_buffer
    end = interval["End Time"] + interval_buffer

    messenger_data = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG"], start, end
    )

    # Get model_ouput between these times
    probabilities = model_output.loc[model_output["Time"].between(start, end)]

    # Search the model output for new crossings in this interval
    new_crossings = hollman_crossings.loc[
        hollman_crossings["Time"].between(start, end)
    ].reset_index(drop=True)

    # Create a figure and plot the mag data
    fig, (region_ax, ax, probability_ax) = plt.subplots(
        3, 1, height_ratios=[1, 30, 10], sharex=True
    )

    ax.plot(
        messenger_data["date"],
        messenger_data["Bx"],
        color=wong_colours["red"],
        lw=1,
        alpha=0.7,
        label="$B_x$",
    )
    ax.plot(
        messenger_data["date"],
        messenger_data["By"],
        color=wong_colours["green"],
        lw=1,
        alpha=0.7,
        label="$B_y$",
    )
    ax.plot(
        messenger_data["date"],
        messenger_data["Bz"],
        color=wong_colours["blue"],
        lw=1,
        alpha=0.7,
        label="$B_z$",
    )
    ax.plot(
        messenger_data["date"],
        messenger_data["|B|"],
        color=wong_colours["black"],
        lw=1,
        label="$|B|$",
    )

    ax.set_ylabel("Magnetic Field Strength [nT]")
    ax.margins(0)
    mag_legend = ax.legend()

    probability_ax.plot(
        probabilities["Time"],
        probabilities["P(SW)"],
        color=wong_colours["yellow"],
        lw=3,
        label="P(SW)",
    )
    probability_ax.plot(
        probabilities["Time"],
        probabilities["P(MSh)"],
        color=wong_colours["orange"],
        lw=3,
        label="P(MSh)",
    )
    probability_ax.plot(
        probabilities["Time"],
        probabilities["P(MSp)"],
        color=wong_colours["blue"],
        lw=3,
        label="P(MSp)",
    )
    probability_ax.legend()
    probability_ax.set_ylim(0, 1)
    plotting.Add_Tick_Ephemeris(probability_ax)

    # Add boundary crossing intervals
    boundaries.Plot_Crossing_Intervals(
        ax,
        interval["Start Time"] - interval_buffer,
        interval["End Time"] + interval_buffer,
        crossing_intervals,
        lw=3,
        color="black",
    )

    # Add Pump crossings
    relevant_pump_crossings = pump_crossings.loc[
        pump_crossings["Time"].between(
            interval["Start Time"] - interval_buffer,
            interval["End Time"] + interval_buffer,
        )
    ]

    pump_colours = ["purple", "green", "red"]
    for _, c in relevant_pump_crossings.iterrows():
        ax.axvline(c["Time"], ls="dotted", color=pump_colours[c["indicator"] - 1])

    # Plot new crossings

    # Text-box formatting
    text_box_formatting = dict(
        boxstyle="square", facecolor="white", edgecolor="black", pad=0, alpha=1
    )

    crossing_labels = []
    for index, c in new_crossings.iterrows():
        ax.axvline(c["Time"], color="black", ls="dashed")

        label_y = np.arange(0, 1, 0.2)[index % 5]
        crossing_label = ax.text(
            c["Time"],
            label_y,
            c["Transition"],
            va="bottom",
            ha="right",
            fontweight="bold",
            fontsize="small",
            rotation=90,
            transform=ax.get_xaxis_transform(),
            bbox=text_box_formatting,
        )
        crossing_labels.append(crossing_label)

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

            region_ax.axvspan(start, c["Time"], color=shade)

            if len(new_crossings) == 1:
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

                region_ax.axvspan(c["Time"], end, color=shade)

        if index < len(new_crossings) - 1:

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

            region_ax.axvspan(
                c["Time"], new_crossings.loc[index + 1]["Time"], color=shade
            )

        elif index == len(new_crossings) - 1:

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

            region_ax.axvspan(c["Time"], end, color=shade)

    fig.subplots_adjust(hspace=0)
    region_ax.set_yticklabels([])
    region_ax.margins(0)

    plt.show()

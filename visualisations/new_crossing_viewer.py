import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import adjustText
from hermpy import boundaries, mag, plotting, utils
from hermpy.plotting import wong_colours

# We want to look randomly at crossing intervals from the Philpott list,
# and see where the new model placed crossings.
crossing_intervals = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=False
)

# Load the new crossing list
hollman_crossings = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_crossings.csv"
)
hollman_crossings["Time"] = pd.to_datetime(hollman_crossings["Time"])

# Randomly sort the intervals and loop through them
random_crossing_intervals = crossing_intervals.sample(frac=1)

for i, interval in random_crossing_intervals.iterrows():

    # Load data around the interval
    interval_buffer = dt.timedelta(minutes=10)

    start = interval["Start Time"] - interval_buffer
    end = interval["End Time"] + interval_buffer

    messenger_data = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG"], start, end
    )

    # Search the model output for new crossings in this interval
    new_crossings = hollman_crossings.loc[hollman_crossings["Time"].between(start, end)].reset_index(drop=True)

    # Create a figure and plot the mag data
    fig, (region_ax, ax) = plt.subplots(2, 1, height_ratios=[1,30], sharex=True)

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
    plotting.Add_Tick_Ephemeris(ax)
    mag_legend = ax.legend()

    """
    # Add boundary crossing intervals
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
            fontsize="large",
            rotation=90,
            transform=ax.get_xaxis_transform(),
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

            region_ax.axvspan(c["Time"], new_crossings.loc[index + 1]["Time"], color=shade)

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

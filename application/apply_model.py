import datetime as dt
import multiprocessing
import pickle
from operator import is_

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.ensemble
from hermpy import boundaries, mag, plotting, trajectory, utils
from skimage.restoration import denoise_tv_chambolle
from tqdm import tqdm

plot_classifications = True
plot_crossings = True

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

crossings = boundaries.Load_Crossings(utils.User.CROSSING_LISTS["Philpott"])

"""
crossings = crossings.loc[
    crossings["Start Time"].between(dt.datetime(2011, 1, 1), dt.datetime(2012, 1, 1))
]
"""

crossings = crossings.sample(frac=1)

# crossings = crossings.loc[crossings["Start Time"].between(dt.datetime(2011, 11, 22, 7), dt.datetime(2011, 11, 22, 9))]

time_buffer = dt.timedelta(minutes=10)

# Import application parameters
window_size = 10  # seconds. How large of a window to feed to the random forest
step_size = 1  # seconds. How far should the window jump each time

smoothing = "None"  # "TVD", "BoxCar", "None"
smoothing_size = 5

remove_smallest_regions = True
region_length_minimum = 5  # times step size

skip_low_success = False
minimum_success_rate = 0.6

# Load Model
print("Loading model")

all_models = {
    "Random Forest": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/model_rf_region_classifier",
    "Stratified Random Forest": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/model_stratified_three_region_classifier",
    "New Random Forest": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/model_new",
    "Multi_Model": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/multi_model_rf",
}

with open(all_models["Multi_Model"], "rb") as file:
    models = pickle.load(file)

model_features = sorted(models[-1].feature_names_in_)


# Function to get window features in parallel
def Get_Window_Features(input):
    window_start, window_end = input

    data_section = data.query("@window_start <= date <= @window_end")

    if data_section.empty:
        return None

    # Find features
    features = dict()
    for component in ["|B|", "Bx", "By", "Bz"]:
        component_data = data_section[component]
        features.update(
            {
                f"Mean {component}": np.mean(component_data),
                f"Median {component}": np.median(component_data),
                f"Standard Deviation {component}": np.std(component_data),
                f"Skew {component}": scipy.stats.skew(component_data),
                f"Kurtosis {component}": scipy.stats.kurtosis(component_data),
            }
        )

    middle_data_point = data_section.iloc[len(data_section) // 2]
    middle_position = [
        middle_data_point["X MSM' (radii)"],
        middle_data_point["Y MSM' (radii)"],
        middle_data_point["Z MSM' (radii)"],
    ]
    middle_features = [
        "X MSM' (radii)",
        "Y MSM' (radii)",
        "Z MSM' (radii)",
    ]
    for feature in middle_features:
        features[feature] = middle_data_point[feature]

    features.update(
        {
            "Latitude (deg.)": trajectory.Latitude(middle_position),
            "Magnetic Latitude (deg.)": trajectory.Magnetic_Latitude(middle_position),
            "Local Time (hrs)": trajectory.Local_Time(middle_position),
            "Heliocentric Distance (AU)": trajectory.Get_Heliocentric_Distance(
                middle_data_point["date"]
            ),
        }
    )

    # Prediction
    X = pd.DataFrame([features]).reindex(columns=model_features, fill_value=0)

    return X


def Multi_Model_Predict(models: list[sklearn.ensemble.RandomForestClassifier], samples):
    """
    Average the predictions from multiple models
    """

    predictions = np.array([model.predict_proba(samples) for model in models])

    probabilities = np.mean(predictions, axis=0)
    probabilities_stds = np.std(predictions, axis=0)

    return probabilities, probabilities_stds


for i, crossing in crossings.iterrows():

    print(f"Processing crossing {i}")
    # Import time series
    start = crossing["Start Time"] - time_buffer
    end = crossing["End Time"] + time_buffer

    print(f"Loading data between {start} and {end}")
    data = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG_FULL"], start, end, average=None
    )

    windows = [
        (window_start, window_start + dt.timedelta(seconds=window_size))
        for window_start in pd.date_range(start=start, end=end, freq=f"{step_size}s")
    ]
    window_centres = [
        window_start + (window_end - window_start) / 2
        for window_start, window_end in windows
    ]

    # We need the end of the windows for a calculation later
    window_ends = [end for _, end in windows]

    samples = []
    missing = []
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        for sample_id, sample in enumerate(
            tqdm(pool.imap(Get_Window_Features, windows), total=len(windows))
        ):

            if sample is not None:
                samples.append(sample)
                missing.append([0, 0, 0])
            else:
                missing.append([1, 1, 1])

    probabilities = np.full((len(windows), 3), np.nan)
    probability_errors = np.full((len(windows), 3), np.nan)

    valid_indices = np.array(missing)[:, 0] == 0

    if samples:  # Check if we have any samples
        samples = pd.concat(samples, ignore_index=True)
        probabilities[valid_indices, :], probability_errors[valid_indices, :] = (
            Multi_Model_Predict(models, samples)
        )

    else:
        raise ValueError("All samples missing")

    fig, axes = plt.subplots(
        3, 1, gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True, figsize=(8, 6)
    )
    (ax, probability_ax, probability_difference_ax) = axes

    ax.plot(
        data["date"],
        data["Bx"],
        color=wong_colours["red"],
        lw=1,
        alpha=0.7,
        label="$B_x$",
    )
    ax.plot(
        data["date"],
        data["By"],
        color=wong_colours["green"],
        lw=1,
        alpha=0.7,
        label="$B_y$",
    )
    ax.plot(
        data["date"],
        data["Bz"],
        color=wong_colours["blue"],
        lw=1,
        alpha=0.7,
        label="$B_z$",
    )
    ax.plot(data["date"], data["|B|"], color=wong_colours["black"], lw=1, label="$|B|$")

    probability_ax.plot(
        window_centres,
        probabilities[:, 0],
        color=wong_colours["orange"],
        lw=2,
        label="Magnetosheath",
    )
    probability_ax.fill_between(
        window_centres,
        probabilities[:, 0] - probability_errors[:, 0],
        probabilities[:, 0] + probability_errors[:, 0],
        color=wong_colours["orange"],
        alpha=0.5,
    )

    probability_ax.plot(
        window_centres,
        probabilities[:, 1],
        color=wong_colours["pink"],
        lw=2,
        label="Magnetosphere",
    )
    probability_ax.fill_between(
        window_centres,
        probabilities[:, 1] - probability_errors[:, 1],
        probabilities[:, 1] + probability_errors[:, 1],
        color=wong_colours["pink"],
        alpha=0.5,
    )

    probability_ax.plot(
        window_centres,
        probabilities[:, 2],
        color=wong_colours["yellow"],
        lw=2,
        label="Solar Wind",
    )
    probability_ax.fill_between(
        window_centres,
        probabilities[:, 2] - probability_errors[:, 2],
        probabilities[:, 2] + probability_errors[:, 2],
        color=wong_colours["yellow"],
        alpha=0.5,
    )

    # Ratio plot
    # Sort the rows:
    sorted_probabilities = np.sort(probabilities, axis=1)
    largest_probabilities = sorted_probabilities[:, -1]
    second_largest_probabities = sorted_probabilities[:, -2]

    probability_difference = np.abs(largest_probabilities - second_largest_probabities)

    probability_difference_ax.plot(
        window_centres, probability_difference, color="black"
    )

    # Moving Average
    moving_average_size = 5
    moving_average = np.convolve(
        probability_difference,
        np.ones(moving_average_size) / moving_average_size,
        mode="valid",
    )
    probability_difference_ax.plot(
        window_centres[(moving_average_size - 1) // 2 : -(moving_average_size // 2)],
        moving_average,
        color=wong_colours["red"],
        label=f"Moving Average order {moving_average_size}",
    )

    # We can also add the average 3 sigma error of all models
    average_probability_error = np.mean(probability_errors, axis=1)

    error_multiplier = 3
    probability_difference_ax.fill_between(
        window_centres,
        0,
        error_multiplier * average_probability_error,
        label=f"${error_multiplier}$" + r"$\times$ Probability Average Standard Deviation",
        color="grey",
        alpha=0.5,
    )

    probability_difference_ax.legend()
    probability_difference_ax.set_yscale("log")
    probability_difference_ax.set_ylabel(
        "Proability Difference\n|most-likely - next-most-likely|"
    )
    probability_difference_ax.set_ylim(0.01, 1)
    probability_difference_ax.margins(0)

    # Classification
    highest_probabilities = np.argmax(probabilities, axis=1)

    is_magnetosheath = highest_probabilities == 0
    is_magnetosphere = highest_probabilities == 1
    is_solar_wind = highest_probabilities == 2

    # Add region shading
    # Iterate through each window
    magnetosheath_labelled = False
    magnetosphere_labelled = False
    solar_wind_labelled = False
    for i in range(len(window_centres) - 1):

        alpha = 0.5

        if is_magnetosheath[i]:
            ax.axvspan(
                window_centres[i] - (dt.timedelta(seconds=step_size)) / 2,
                window_centres[i] + (dt.timedelta(seconds=step_size)) / 2,
                facecolor=wong_colours["orange"],
                edgecolor=None,
                alpha=alpha,
                label=(
                    "Prediction = Magnetosheath" if not magnetosheath_labelled else ""
                ),
            )
            magnetosheath_labelled = True

        elif is_magnetosphere[i]:
            ax.axvspan(
                window_centres[i] - (dt.timedelta(seconds=step_size)) / 2,
                window_centres[i] + (dt.timedelta(seconds=step_size)) / 2,
                facecolor=wong_colours["pink"],
                edgecolor=None,
                alpha=alpha,
                label=(
                    "Prediction = Magnetosphere" if not magnetosphere_labelled else ""
                ),
            )
            magnetosphere_labelled = True

        elif is_solar_wind[i]:
            ax.axvspan(
                window_centres[i] - (dt.timedelta(seconds=step_size)) / 2,
                window_centres[i] + (dt.timedelta(seconds=step_size)) / 2,
                facecolor=wong_colours["yellow"],
                edgecolor=None,
                alpha=alpha,
                label="Prediction = Solar Wind" if not solar_wind_labelled else "",
            )
            solar_wind_labelled = True

    probability_ax.legend()

    mag_legend = ax.legend()

    ax.set_ylabel("Magnetic Field Strength [nT]")
    probability_ax.set_ylabel(f"Region Probability")

    ax.set_title(
        f"Model Application (Overlapping Sliding Window)\nWindow Size: {window_size} s    Step Size: {step_size} s"
    )

    ax.margins(0)
    probability_ax.margins(0)
    fig.subplots_adjust(hspace=0)
    plotting.Add_Tick_Ephemeris(ax)

    probability_ax.set_ylim(0, 1)

    # Add boundary crossing intervals
    boundaries.Plot_Crossing_Intervals(probability_ax, start, end, crossings, lw=3, color="black")

    # FIND CROSSINGS
    region_labels = np.empty_like(window_centres)
    region_labels[is_magnetosheath] = "Magnetosheath"
    region_labels[is_magnetosphere] = "Magnetosphere"
    region_labels[is_solar_wind] = "Solar Wind"

    new_crossings = np.where(region_labels[:-1] != region_labels[1:])[0]

    if len(new_crossings) != 0:

        new_crossings_info: list[dict] = []
        for crossing_index in new_crossings:
            # Crossings must be inside the moving average limit to be considered
            if (crossing_index < (moving_average_size - 1) // 2) or (crossing_index > len(moving_average)):
                continue

            crossing_time = (
                window_centres[crossing_index]
                + (window_centres[crossing_index + 1] - window_centres[crossing_index]) / 2
            )
            region_before = region_labels[crossing_index]
            region_after = region_labels[crossing_index + 1]

            new_crossings_info.append(
                {
                    "Time": crossing_time,
                    "Region Before": region_before,
                    "Region After": region_after,
                }
            )

        new_crossings_info = pd.DataFrame.from_dict(new_crossings_info)

        # Remove crossings too close together
        drop_mask = np.array([False] * len(new_crossings_info))

        # Check time differences
        threshold = dt.timedelta(seconds=1)

        for i in range(1, len(new_crossings_info)):

            if (new_crossings_info["Time"].iloc[i] - new_crossings_info["Time"].iloc[i - 1]) <= threshold:
                drop_mask[i] = True
                drop_mask[i-1] = True

        cleaned_crossings = new_crossings_info[~drop_mask]

        for ax in axes:
            for t in cleaned_crossings["Time"]:
                ax.axvline(t, color="black", ls="dotted")

        print(new_crossings_info)
        print(cleaned_crossings)

    plt.tight_layout()
    plt.show()

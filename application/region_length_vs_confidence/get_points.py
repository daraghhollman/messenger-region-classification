import datetime as dt
import multiprocessing
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.ensemble
from hermpy import boundaries, mag, plotting, trajectory, utils
from tqdm import tqdm

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

crossings = crossings.sample(
    frac=0.002
)  # only take values for a small number of crossings

print(f"Processing {len(crossings)} crossings")

time_buffer = dt.timedelta(minutes=10)

# Import application parameters
window_size = 10  # seconds. How large of a window to feed to the random forest
step_size = 1  # seconds. How far should the window jump each time

# Load Model
print("Loading model")

all_models = {
    "Random Forest": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/model_rf_region_classifier",
    "Stratified Random Forest": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/model_stratified_three_region_classifier",
    "New Random Forest": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/model_new",
    "Multi_Model_RF": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/multi_model_rf",
    "Multi_Model_HGB": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/multi_model_hgb",
}

with open(all_models["Multi_Model_RF"], "rb") as file:
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


all_regions = []
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

    # Ratios
    # Sort the rows:
    sorted_probabilities = np.sort(probabilities, axis=1)
    largest_probabilities = sorted_probabilities[:, -1]
    second_largest_probabities = sorted_probabilities[:, -2]

    probability_difference = np.abs(largest_probabilities - second_largest_probabities)
    probability_ratio = second_largest_probabities / largest_probabilities

    # Classification
    highest_probabilities = np.argmax(probabilities, axis=1)

    is_magnetosheath = highest_probabilities == 0
    is_magnetosphere = highest_probabilities == 1
    is_solar_wind = highest_probabilities == 2

    region_labels = np.empty_like(window_centres)
    region_labels[is_magnetosheath] = "Magnetosheath"
    region_labels[is_magnetosphere] = "Magnetosphere"
    region_labels[is_solar_wind] = "Solar Wind"

    # Define regions
    crossing_indices = np.where(region_labels[:-1] != region_labels[1:])[0]

    # N = 0 is not possible as we centre on a crossing interval
    # N = 1 we can't determine metrics for as region duration is undefined
    # N = 2 is non-physical as we must have an odd number of crossings
    #       for a boundary crossing
    if len(crossing_indices) < 3:
        print("Not enough crossings, skipping")
        continue

    regions = []
    # Exclude the last region as i+1 is not defined
    for i in range(len(crossing_indices[:-1])):
        current_crossing_index = crossing_indices[i]
        next_crossing_index = crossing_indices[i + 1]
        middle_index = current_crossing_index + int(
            (next_crossing_index - current_crossing_index) / 2
        )
        regions.append(
            {
                "Start Time": window_centres[current_crossing_index],
                "End Time": window_centres[next_crossing_index],
                "Duration (seconds)": (
                    window_centres[next_crossing_index]
                    - window_centres[current_crossing_index]
                ).total_seconds(),
                "Label": region_labels[middle_index],
                # Including the values at the crossing points
                "Confidence (Mean Method)": 1
                - np.mean(
                    probability_ratio[current_crossing_index : next_crossing_index + 1]
                ),
                "Confidence (Median Method)": 1
                - np.median(
                    probability_ratio[current_crossing_index : next_crossing_index + 1]
                ),
            }
        )

    # As the beginning and end regions start and finish arbitrarly,
    # we exclude them from the metrics.
    all_regions.extend(regions[1:-1])

region_data = pd.DataFrame(all_regions)

region_data.to_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/application/region_length_vs_confidence/region_metrics.csv"
)

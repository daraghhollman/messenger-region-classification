import contextlib
import datetime as dt
import pickle

import joblib
import numpy as np
import pandas as pd
from hermpy import boundaries, mag, trajectory, utils
from tqdm import tqdm

n_jobs = -1

# Load Model
model_path = "../models/model"
print(f"Loading model from {model_path}")
with open(
    model_path,
    "rb",
) as file:
    model = pickle.load(file)
model_features = sorted(model.feature_names_in_)


# Load crossings
print(f"Loading crossing intervals from {utils.User.CROSSING_LISTS['Philpott']}")
crossings = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"], include_data_gaps=True
)

# To ensure no overlap, we want to classify pairs of crossings as one.
# i.e. BS_IN and MP_IN, as well as MP_OUT and BS_OUT
# However, the are sometimes missing crossings, so we need need to be careful
# Based on the geometry of the orbit and the physics of the system,
# we never expect to see MP_IN closely followed by any BS crossing.
# Similarly, we never expect to see BS_OUT closely followed by any MP crossing.

print("Grouping Crossings")
crossing_groups = []

crossing_index = 0
while crossing_index < len(crossings) - 1:

    current_crossing = crossings.loc[crossing_index]
    next_crossing = crossings.loc[crossing_index + 1]

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

# Now that we have a list containing all the groups of crossing intervals we want
# to search around. We can loop through this (with multiprocessing) and find our
# individual crossings.

# Parameters of the application
# How much to buffer each group
time_buffer = dt.timedelta(minutes=10)

window_size = 10  # seconds. How large of a window to pass to the model at a time
step_size = 1  # seconds. How far should the window jump each time


def Get_Window_Features(data, window_start, window_end):

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
            "Mid-Time": middle_data_point["date"],
            "Latitude (deg.)": trajectory.Latitude(middle_position),
            "Magnetic Latitude (deg.)": trajectory.Magnetic_Latitude(middle_position),
            "Local Time (hrs)": trajectory.Local_Time(middle_position),
        }
    )

    # Only include features which are in the model
    X = pd.DataFrame([features])

    return X


def Get_Probabilities(crossing_interval_group):

    # if group is a pair
    # set data start time to start of first interval -

    # Check if crossing group is a pair or individual
    group_is_pair = True

    if len(crossing_interval_group) == 1:
        group_is_pair = False

    if group_is_pair:
        data_start_time = crossing_interval_group[0]["Start Time"] - time_buffer
        data_end_time = crossing_interval_group[1]["End Time"] + time_buffer

    else:
        data_start_time = crossing_interval_group[0]["Start Time"] - time_buffer
        data_end_time = crossing_interval_group[0]["End Time"] + time_buffer

    # Load high resolution mag data
    # This data is aberrated
    data = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG_FULL"],
        data_start_time,
        data_end_time,
        average=None,
    )

    # Create windows to classify
    windows = [
        (window_start, window_start + dt.timedelta(seconds=window_size))
        for window_start in pd.date_range(
            start=data_start_time, end=data_end_time, freq=f"{step_size}s"
        )
    ]
    window_centres = [
        window_start + (window_end - window_start) / 2
        for window_start, window_end in windows
    ]

    samples = []
    missing = []

    # Get features from each window
    for window_start, window_end in windows:

        sample = Get_Window_Features(data, window_start, window_end)

        if sample is not None:
            samples.append(sample)
            missing.append([0, 0, 0])
        else:
            missing.append([1, 1, 1])

    probabilities = np.full((len(windows), 3), np.nan)

    valid_indices = np.array(missing)[:, 0] == 0

    if samples:  # Check if we have any samples
        # Then make predictions!
        samples = pd.concat(samples, ignore_index=True)

        samples["Heliocentric Distance (AU)"] = utils.Constants.KM_TO_AU(
            trajectory.Get_Heliocentric_Distance(
                samples["Mid-Time"].to_list()
            )
        )

        # Ensure everything is in the correct order
        samples = samples.reindex(columns=model_features)

        predictions = model.predict_proba(samples)
        probabilities[valid_indices, :] = predictions

    else:
        raise ValueError("All samples missing")

    # Now we have three lists:
    # window_centres        (1 x N) - The time stamp of our probabilities
    # probabilities         (3 x N) - The average probability of classification for each class over 10 models

    return window_centres, probabilities


# "Tracking progress of joblib.Parallel execution"
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


with tqdm_joblib(
    tqdm(desc="Applying model to crossing intervals", dynamic_ncols=True, smoothing=0, total=len(crossing_groups))
) as progress_bar:
    results = joblib.Parallel(n_jobs=n_jobs, temp_folder="/net/romulus.ap.dias.ie/romulus/dhollman/tmp/")(
        joblib.delayed(Get_Probabilities)(group) for group in crossing_groups
    )


times, probabilities = zip(*results)  # Unpack results

# Convert lists of times and probabilities into arrays
times = np.concatenate(times)
probabilities = np.vstack(probabilities)

data_to_save = {
    "Time": times,
    "P(MSh)": probabilities[:, 0],
    "P(MSp)": probabilities[:, 1],
    "P(SW)": probabilities[:, 2],
}

pd.DataFrame(data_to_save).to_csv("../outputs/model_ouput.csv", index=False)

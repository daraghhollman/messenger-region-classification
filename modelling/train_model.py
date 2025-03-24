"""
A script to train and save a decision tree based model as a region classifier for
solar wind, magnetosheath, and magnetosphere.
"""

import pickle

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from tqdm import tqdm

SEED = 0
METHOD = "RF"  # RF, GB, HGB
model_params = {"n_estimators": 100, "max_depth": 15, "max_features": "sqrt"}

print(model_params)


def main():
    # Load extracted features
    inputs = {
        "Solar Wind": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/solar_wind_samples.csv",
        "BS Magnetosheath": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/bs_magnetosheath_samples.csv",
        "MP Magnetosheath": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/mp_magnetosheath_samples.csv",
        "Magnetopause": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/magnetosphere_samples.csv",
    }

    print("Loading data")
    features_data = []
    for region_type in inputs.keys():
        features_data.append(pd.read_csv(inputs[region_type]))

    print("Data loaded")

    # Combine into one dataframe
    features_data = pd.concat(features_data, ignore_index=True).dropna()

    print(f"Length: {len(features_data)}")

    features_data = features_data.drop(
        columns=["Sample Start", "Sample End", "Boundary ID"]
    )

    print("Formatting columns")
    expanded_feature_labels = ["|B|", "Bx", "By", "Bz"]
    for feature in ["Mean", "Median", "Standard Deviation", "Skew", "Kurtosis"]:

        # Convert elements from list-like strings to lists of floats
        features_data[feature] = features_data[feature].apply(
            lambda s: list(map(float, s.strip("[]").split()))
        )

        # Expand feature lists into new columns
        expanded_columns = (
            features_data[feature]
            .apply(pd.Series)
            .rename(lambda x: f"{feature} {expanded_feature_labels[x]}", axis=1)
        )

        # Assign new columns back to the original dataset
        features_data[expanded_columns.columns] = expanded_columns

    features_data = features_data.drop(
        columns=["Mean", "Median", "Standard Deviation", "Skew", "Kurtosis"]
    )

    # Feature Removal
    features_data = features_data.drop(
        columns=[
            "Skew |B|",
            "Skew Bx",
            "Skew By",
            "Skew Bz",
            "Kurtosis |B|",
            "Kurtosis Bx",
            "Kurtosis By",
            "Kurtosis Bz",
        ]
    )

    # Try with and without bs and mp magnetosheath as separate features
    features_data["Label"] = features_data["Label"].replace(
        "BS Magnetosheath", "Magnetosheath"
    )
    features_data["Label"] = features_data["Label"].replace(
        "MP Magnetosheath", "Magnetosheath"
    )
    # Put some of the data to the side which we can test on
    # after tuning our parameters.
    evaluation_data = features_data.sample(frac=0.2, random_state=SEED)
    features_data = features_data.drop(evaluation_data.index)

    column_names = list(features_data.columns.values)
    column_names.sort()
    evaluation_X = evaluation_data[column_names]
    evaluation_X = evaluation_X.drop(columns=["Label"])

    evaluation_y = evaluation_data["Label"]  # Target

    print(
        f"Isolated 20% of data for true testing. New data length {len(features_data)}"
    )

    column_names = list(features_data.columns.values)
    column_names.sort()
    X = features_data[column_names]
    X = X.drop(columns=["Label"])

    y = features_data["Label"]  # Target

    # MODELLING #
    print("Beginning Modeling")
    num_models = 10
    kfold = StratifiedKFold(n_splits=num_models, shuffle=True, random_state=SEED)

    models = []
    accuracies = []
    training_accuracies = []
    confusion_matrices = []

    for train_index, test_index in tqdm(
        kfold.split(X, y), desc="Training Model", total=num_models
    ):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        current_model = RandomForestClassifier(
            **model_params, random_state=SEED, n_jobs=1
        )

        current_model.fit(X_train, y_train)

        y_pred = current_model.predict(X_test)
        y_pred_training = current_model.predict(X_train)

        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, y_pred_training)

        models.append(current_model)
        accuracies.append(test_accuracy)
        training_accuracies.append(train_accuracy)
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    # Compute average accuracy across all folds
    avg_test_accuracy = np.mean(accuracies)
    avg_test_accuracy_error = np.std(accuracies)
    avg_train_accuracy = np.mean(training_accuracies)
    avg_train_accuracy_error = np.std(training_accuracies)
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    confusion_matrix_std = np.std(confusion_matrices, axis=0)

    print(
        f"\nAverage Test Accuracy: {avg_test_accuracy:.5f} +/- {avg_test_accuracy_error:.5f}"
    )
    print(
        f"Average Training Accuracy: {avg_train_accuracy:.8f} +/- {avg_train_accuracy_error:.8f}"
    )

    if METHOD == "RF":
        importances = np.array([model.feature_importances_ for model in models]).T
        mean_importances = np.mean(importances, axis=1)
        feature_names = X.columns

        # Sorting
        sorted_indices = np.argsort(mean_importances)[::-1]
        importances = importances[sorted_indices, :]  # Reorder importances
        mean_importances = mean_importances[sorted_indices]  # Reorder mean values
        feature_names = [
            feature_names[i] for i in sorted_indices
        ]  # Reorder feature names

        # Create a boxplot
        plt.figure(figsize=(12, 6))

        y_positions = np.arange(len(feature_names))

        plt.barh(y_positions, mean_importances, color="white", edgecolor="black")
        sns.boxplot(data=importances.T, orient="h", color="indianred")

        plt.yticks(
            y_positions, feature_names
        )  # Ensure feature names are correctly positioned

        # Add feature names as labels
        plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
        plt.xlabel("Feature Importance")
        plt.title(f"Feature Importance Distribution Across {num_models} Folds")

    print("\nConfusion Matrix\n")
    print(avg_confusion_matrix)

    # Plot the average confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        avg_confusion_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar=True,
        xticklabels=["Magnetosheath", "Magnetosphere", "Solar Wind"],
        yticklabels=["Magnetosheath", "Magnetosphere", "Solar Wind"],
        norm=matplotlib.colors.LogNorm(),
    )
    # plt.title("Average Confusion Matrix with Std")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Annotate with standard deviation
    for i in range(avg_confusion_matrix.shape[0]):
        for j in range(avg_confusion_matrix.shape[1]):
            plt.text(
                j + 0.5,
                i + 0.65,
                f"±{confusion_matrix_std[i, j]:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black" if i == j else "white",
            )

    plt.show()

    #### EVALUTATION ####
    evaluation_predictions = [m.predict(evaluation_X) for m in models]
    evaluation_confusion_matrices = [
        confusion_matrix(evaluation_y, predictions)
        for predictions in evaluation_predictions
    ]

    print(
        f"Evaluation Accuracy: {np.mean([accuracy_score(evaluation_y, predictions) for predictions in evaluation_predictions])} +/- {np.std([accuracy_score(evaluation_y, predictions) for predictions in evaluation_predictions])}"
    )

    mean_evaluation_confusion_matrix = np.mean(evaluation_confusion_matrices, axis=0)
    std_evaluation_confusion_matrix = np.std(evaluation_confusion_matrices, axis=0)

    # Plot the average confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mean_evaluation_confusion_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar=True,
        xticklabels=["Magnetosheath", "Magnetosphere", "Solar Wind"],
        yticklabels=["Magnetosheath", "Magnetosphere", "Solar Wind"],
        norm=matplotlib.colors.LogNorm(),
    )
    # plt.title("Average Confusion Matrix with Std")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Annotate with standard deviation
    for i in range(mean_evaluation_confusion_matrix.shape[0]):
        for j in range(mean_evaluation_confusion_matrix.shape[1]):
            plt.text(
                j + 0.5,
                i + 0.65,
                f"±{std_evaluation_confusion_matrix[i, j]:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black" if i == j else "white",
            )

    plt.show()

    match METHOD:
        case "RF":
            output_file = "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/multi_model_rf"

        case "GB":
            output_file = "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/multi_model_gb"

        case "HGB":
            output_file = "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/multi_model_hgb"

        case _:
            raise ValueError(f"Invalid modelling method '{METHOD}'")

    with open(
        output_file,
        "wb",
    ) as file:
        pickle.dump(models, file)


def Show_Training_Spread(training_data):
    """A function to check if the training data is disperse spatially.

    This is done by plotting distributions of spatial features.

    """

    features_to_test = [
        "Heliocentric Distance (AU)",
        "Local Time (hrs)",
        "Latitude (deg.)",
        "Magnetic Latitude (deg.)",
        "X MSM' (radii)",
        "Y MSM' (radii)",
        "Z MSM' (radii)",
    ]

    for feature in features_to_test:
        _, ax = plt.subplots()

        ax.hist(training_data[feature], color="black")

        ax.set_xlabel(feature)
        ax.set_ylabel("# Events")

        ax.margins(0)

        plt.show()


if __name__ == "__main__":
    main()

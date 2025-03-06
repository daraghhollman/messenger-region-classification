"""
A script to train and save a decision tree based model as a region classifier for
solar wind, magnetosheath, and magnetosphere.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

SEED = 0


def main():
    # Load extracted features
    inputs = {
        "Solar Wind": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/solar_wind_samples.csv",
        "BS Magnetosheath": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/bs_magnetosheath_samples.csv",
        "MP Magnetosheath": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/mp_magnetosheath_samples.csv",
        "Magnetopause": "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/magnetosphere_samples.csv",
    }

    features_data = []
    for region_type in inputs.keys():
        features_data.append(pd.read_csv(inputs[region_type]))

    # Combine into one dataframe
    features_data = pd.concat(features_data, ignore_index=True).dropna()

    features_data = features_data.drop(
        columns=["Sample Start", "Sample End", "Boundary ID"]
    )

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

    # Try with and without bs and mp magnetosheath as separate features
    features_data["Label"] = features_data["Label"].replace(
        "BS Magnetosheath", "Magnetosheath"
    )
    features_data["Label"] = features_data["Label"].replace(
        "MP Magnetosheath", "Magnetosheath"
    )

    column_names = list(features_data.columns.values)
    column_names.sort()
    X = features_data[column_names]
    X = X.drop(columns=["Label"])

    y = features_data["Label"]  # Target

    # MODELLING #
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

        random_forest = RandomForestClassifier(n_estimators=100, random_state=SEED)
        random_forest.fit(X_train, y_train)

        y_pred = random_forest.predict(X_test)
        y_pred_training = random_forest.predict(X_train)

        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, y_pred_training)

        models.append(random_forest)
        accuracies.append(test_accuracy)
        training_accuracies.append(train_accuracy)
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    # Compute average accuracy across all folds
    avg_test_accuracy = np.mean(accuracies)
    avg_train_accuracy = np.mean(training_accuracies)
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    confusion_matrix_std = np.std(confusion_matrices, axis=0)

    print(f"\nAverage Test Accuracy: {avg_test_accuracy:.5f}")
    print(f"Average Training Accuracy: {avg_train_accuracy:.5f}")

    # The following will simply use the last random forest model which was generated

    """
    if input("Show training spread? [y/N]\n > ") == "y":
        Show_Training_Spread(X_train)
    """

    importances = np.array([model.feature_importances_ for model in models]).T
    mean_importances = np.mean(importances, axis=1)
    feature_names = X.columns

    # Sorting
    sorted_indices = np.argsort(mean_importances)[::-1]
    importances = importances[sorted_indices, :]  # Reorder importances
    mean_importances = mean_importances[sorted_indices]  # Reorder mean values
    feature_names = [feature_names[i] for i in sorted_indices]  # Reorder feature names

    # Create a boxplot
    plt.figure(figsize=(12, 6))

    y_positions = np.arange(len(feature_names))

    plt.barh(y_positions, mean_importances, color="white", edgecolor="black")
    sns.boxplot(data=importances.T, orient="h", color="indianred")

    plt.yticks(y_positions, feature_names)  # Ensure feature names are correctly positioned

    # Add feature names as labels
    plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
    plt.xlabel("Feature Importance")
    plt.title(f"Feature Importance Distribution Across {num_models} Folds")

    print("\nConfusion Matrix\n")
    print(avg_confusion_matrix)

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

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
                color="black" if ((i == 0) and (j == 0)) else "white",
            )

    plt.show()

    with open(
        "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/multi_model_rf",
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

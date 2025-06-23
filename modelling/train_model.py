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
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection


def main():

    # Load extracted features
    # These are in 4 different data sets which we need to combine
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
    # These features were extracted, but are unimportant
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

    # We previously extracted BS and MP adjacent magnetosheath data as different
    # We change their labels to be the same now.
    # This introduces a class imbalance which we must account for.
    features_data["Label"] = features_data["Label"].replace(
        "BS Magnetosheath", "Magnetosheath"
    )
    features_data["Label"] = features_data["Label"].replace(
        "MP Magnetosheath", "Magnetosheath"
    )

    features_data.to_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/training_data.csv"
    )

    # Downsample magnetosheath until classes are balanced
    # Functionally just halving the number of magnetosheath samples
    magnetosheath_rows = features_data.loc[features_data["Label"] == "Magnetosheath"]
    rows_to_remove = magnetosheath_rows.sample(frac=0.5)

    features_data = features_data.drop(rows_to_remove.index)

    # Filter outliers
    extreme_rows = features_data.loc[
        (features_data["Mean |B|"] >= 5000)
        | (features_data["Standard Deviation |B|"] >= 5000)
    ]
    features_data = features_data.drop(extreme_rows.index)
    print(f"Removed {len(extreme_rows)} extreme rows")

    print(
        f"""
        Training data:
        Size: {len(features_data)}
            SW: {len(features_data.loc[features_data["Label"] == "Solar Wind"])}
            MSh: {len(features_data.loc[features_data["Label"] == "Magnetosheath"])}
            MSp: {len(features_data.loc[features_data["Label"] == "Magnetosphere"])}
        """
    )

    # Start our multi-model loop:
    number_of_iterrations = 10
    print(f"Training {number_of_iterrations} models")
    models = []
    testing_accuracies = []
    testing_recalls = []
    testing_precisions = []
    testing_confusion_matrices = []
    for model_index in range(number_of_iterrations):

        # Create a copy of the data
        model_data = features_data.copy()

        # We sort the columns by feature name as this is useful later.
        # When applying the model, the data input must be in the same order.
        X = model_data.drop(columns=["Label"])
        y = model_data["Label"]

        # Sort the column names after dropping the label
        column_names = list(X.columns)
        column_names.sort()
        X = X[column_names]  # Ensure X is ordered by sorted column names

        # MODELLING
        training_X, testing_X, training_y, testing_y = (
            sklearn.model_selection.train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
            )
        )

        print(len(training_X))
        print(len(testing_X))

        # Define our model using default parameters
        model = sklearn.ensemble.RandomForestClassifier(n_jobs=-1)
        model.fit(training_X, training_y)

        models.append(model)

        # Test the model using the testing set
        testing_predictions = model.predict(testing_X)
        testing_accuracy = sklearn.metrics.accuracy_score(
            testing_y, testing_predictions
        )
        testing_recall = sklearn.metrics.recall_score(
            testing_y, testing_predictions, average="micro"
        )
        testing_precision = sklearn.metrics.precision_score(
            testing_y, testing_predictions, average="micro"
        )
        testing_accuracies.append(testing_accuracy)
        testing_recalls.append(testing_recall)
        testing_precisions.append(testing_precision)

        testing_confusion_matrix = sklearn.metrics.confusion_matrix(
            testing_y, testing_predictions
        )
        testing_confusion_matrices.append(testing_confusion_matrix)

        print(f"Model trained with testing accuracy: {testing_accuracy}")

    # Print stats
    print("Accuracy")
    print(f"Mean model testing accuracy: {np.mean(testing_accuracies)}")
    print(f"Std model testing accuracy: {np.std(testing_accuracies)}")

    print("Recall")
    print(f"Mean model testing recall: {np.mean(testing_recalls)}")
    print(f"Std model testing recall: {np.std(testing_recalls)}")

    print("Precision:")
    print(f"Mean model testing precision: {np.mean(testing_precisions)}")
    print(f"Std model testing precision: {np.std(testing_precisions)}")

    # Make plots!
    # Save out confusion matrices, testing accuracies, and models
    # to plot later.
    with open(
        "/home/daraghhollman/Main/Work/papers/boundaries/resources/models",
        "wb",
    ) as file:
        pickle.dump(models, file)

    with open(
        "/home/daraghhollman/Main/Work/papers/boundaries/resources/testing_confusion_matrices",
        "wb",
    ) as file:
        pickle.dump(testing_confusion_matrices, file)

    with open(
        "/home/daraghhollman/Main/Work/papers/boundaries/resources/testing_accuracies",
        "wb",
    ) as file:
        pickle.dump(testing_accuracies, file)

    # AVERAGE FEATURE IMPORTANCE
    importances = np.array([model.feature_importances_ for model in models]).T
    mean_importances = np.mean(importances, axis=1)

    # Sorting
    sorted_indices = np.argsort(mean_importances)[::-1]
    importances = importances[sorted_indices, :]  # Reorder importances
    mean_importances = mean_importances[sorted_indices]  # Reorder mean values
    feature_names = [column_names[i] for i in sorted_indices]  # Reorder feature names

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
    plt.title(f"Average Feature Importance Distribution for {len(models)} models")

    print("\nConfusion Matrix\n")
    avg_confusion_matrix = np.mean(testing_confusion_matrices, axis=0)
    confusion_matrix_std = np.std(testing_confusion_matrices, axis=0)
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

    # If happy with the output, we can save the model
    output_file = "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/model"

    # We arbitraraly save the 1st model
    # Due to the use of random seed, this is functionally a random choice
    with open(
        output_file,
        "wb",
    ) as file:
        pickle.dump(models[0], file)

    print("Model saved to:\n" + output_file)


if __name__ == "__main__":
    main()

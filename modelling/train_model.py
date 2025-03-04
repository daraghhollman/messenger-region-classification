"""
A script to train and save a decision tree based model as a region classifier for
solar wind, magnetosheath, and magnetosphere.
"""

import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify = y
    )

    if input("Show training spread? [y/N]\n > ") == "y":
        Show_Training_Spread(X_train)

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)

    y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    importances = random_forest.feature_importances_
    feature_names = X.columns

    "Features:"
    print(feature_names)

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importance")
    plt.show()

    cm = confusion_matrix(y_test, y_pred, )

    print("\nConfusion Matrix\n")
    print(cm)

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    cm_display = ConfusionMatrixDisplay(
        cm, display_labels=["Magnetosheath", "Magnetosphere", "Solar Wind"]
    )
    cm_display.plot()
    plt.show()

    with open(
        "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/modelling/model_" + input("Save model as: "),
        "wb",
    ) as file:
        pickle.dump(random_forest, file)


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

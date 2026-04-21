import os
import glob
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_data(folder_path):
    print("Loading all CSV files...")

    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []

    for file in all_files:
        print(f"Reading: {os.path.basename(file)}")
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows after merge: {len(full_df)}")

    return full_df


def clean_data(df):
    print("Cleaning dataset...")
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def group_labels(df):
    print("Grouping attack labels...")

    df["Label"] = df["Label"].str.strip()

    attack_mapping = {
        "BENIGN": "BENIGN",

        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "DoS Slowloris": "DoS",
        "DoS Slowhttptest": "DoS",

        "DDoS": "DDoS",
        "PortScan": "PortScan",

        "FTP-Patator": "BruteForce",
        "SSH-Patator": "BruteForce",

        "Web Attack – Brute Force": "WebAttack",
        "Web Attack – XSS": "WebAttack",
        "Web Attack – Sql Injection": "WebAttack",

        "Bot": "Bot",
        "Infiltration": "Infiltration"
    }

    df["Label"] = df["Label"].map(attack_mapping)
    df = df.dropna(subset=["Label"])

    print("Class distribution:")
    print(df["Label"].value_counts())

    return df


def train():

    df = load_data("../data/raw")
    df = clean_data(df)
    df = group_labels(df)

    # Balance only BENIGN
    benign_df = df[df["Label"] == "BENIGN"].sample(n=100000, random_state=42)
    attack_df = df[df["Label"] != "BENIGN"]
    df = pd.concat([benign_df, attack_df])

    print("Balanced distribution:")
    print(df["Label"].value_counts())

    # Encode labels
    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])

    y = df["Label"]

    # Keep only numeric features
    X = df.drop("Label", axis=1)
    X = X.select_dtypes(include=[np.number])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)

    # Train full model
    model_full = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model_full.fit(X_train_scaled, y_train)

    # Feature importance
    importances = pd.Series(
        model_full.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    top_features = importances.head(35).index.tolist()

    print("\nTop 35 Features:")
    print(top_features)

    # Retrain using top 35
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    scaler_final = StandardScaler()
    X_train_top_scaled = scaler_final.fit_transform(X_train_top)
    X_test_top_scaled = scaler_final.transform(X_test_top)

    model_final = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model_final.fit(X_train_top_scaled, y_train)

    y_pred = model_final.predict(X_test_top_scaled)

    print("\nFinal Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs("../models", exist_ok=True)

    joblib.dump(model_final, "../models/final_model.pkl")
    joblib.dump(scaler_final, "../models/scaler.pkl")
    joblib.dump(label_encoder, "../models/label_encoder.pkl")
    joblib.dump(top_features, "../models/selected_features.pkl")

    print("\nModel saved successfully.")


if __name__ == "__main__":
    train()
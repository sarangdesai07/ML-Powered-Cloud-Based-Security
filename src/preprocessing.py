import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ---------------- LOAD DATA ----------------
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


# ---------------- CLEAN DATA ----------------
def clean_data(df):
    print("Cleaning dataset...")

    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


# ---------------- GROUP LABELS ----------------
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

    print("Class distribution before balancing:")
    print(df["Label"].value_counts())

    return df


# ---------------- BALANCE DATA ----------------
def balance_data(df):
    print("Balancing dataset...")

    groups = df.groupby("Label")
    max_samples = 50000  # adjust if needed

    balanced_df = []

    for label, group in groups:
        if len(group) > max_samples:
            balanced_df.append(group.sample(max_samples, random_state=42))
        else:
            balanced_df.append(group)

    df_balanced = pd.concat(balanced_df)

    print("New class distribution:")
    print(df_balanced["Label"].value_counts())

    return df_balanced


# ---------------- ENCODE LABELS ----------------
def encode_labels(df):
    print("Encoding labels...")

    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])

    return df, le
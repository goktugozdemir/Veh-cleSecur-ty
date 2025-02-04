import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
from imblearn.over_sampling import SMOTE  # Import SMOTE

def load_veremi_dataset(veremi_path):
    """
    Reads VeReMi JSON files, extracts features and labels,
    and returns a DataFrame. Assumes each file contains multiple
    JSON objects separated by newlines or not in a JSON array.
    """
    data = []

    for root, dirs, files in os.walk(veremi_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            json_obj = json.loads(line)

                            # Extract relevant data
                            sender_id = json_obj.get("rcvTime", -1)
                            position = json_obj.get("pos", [])
                            speed = json_obj.get("spd", [])
                            acl = json_obj.get("acl", [])
                            heading = json_obj.get("hed", [])

                            label = 1 if "A1" in file else 0

                            data.append({
                                "rcvTime": sender_id,
                                "position_x": position[0] if len(position) > 0 else 0,
                                "position_y": position[1] if len(position) > 1 else 0,
                                "position_z": position[2] if len(position) > 2 else 0,
                                "speed_x": speed[0] if len(speed) > 0 else 0,
                                "speed_y": speed[1] if len(speed) > 1 else 0,
                                "speed_z": speed[2] if len(speed) > 2 else 0,
                                "acl_x": acl[0] if len(acl) > 0 else 0,
                                "acl_y": acl[1] if len(acl) > 1 else 0,
                                "acl_z": acl[2] if len(acl) > 2 else 0,
                                "heading_x": heading[0] if len(heading) > 0 else 0,
                                "heading_y": heading[1] if len(heading) > 1 else 0,
                                "heading_z": heading[2] if len(heading) > 2 else 0,
                                "label": label
                            })
                        except json.JSONDecodeError:
                            print(f"Skipping invalid line in {file}: {line}")

    return pd.DataFrame(data)


def load_vanet_dataset(vanet_file_path):
    """
    Loads the VANET dataset from a CSV file and assigns column names.
    """
    # Define column names based on the dataset's structure
    column_names = [
        "Start_time", "End_time", "Time_Period", "Packets",
        "Rate", "Sender_Stopping_Distance",
        "Receiver_Stopping_Distance", "Actual_Distance", "Severity"
    ]

    # Load the dataset and assign column names
    vanet_data = pd.read_csv(vanet_file_path, header=None, names=column_names)

    # Add a label column (set all rows to '0' for normal class)
    vanet_data['label'] = 0

    return vanet_data

def show_first_10_lines(veremi_data, vanet_data):
    """
    Shows the first 10 rows of both datasets with horizontal lines and waits for button press to continue.
    """

    def on_button_click():
        root.quit()  # Close the Tkinter window after the button press

    root = tk.Tk()
    root.title("Dataset Overview")

    # Display the first 10 rows of each dataset
    veremi_preview = veremi_data.head(10).to_string(index=False)
    vanet_preview = vanet_data.head(10).to_string(index=False)

    # Display datasets in text boxes
    tk.Label(root, text="VeReMi Dataset Preview").pack()

    # Add horizontal lines between rows in VeReMi dataset preview
    veremi_lines = veremi_preview.split('\n')
    for line in veremi_lines:
        tk.Label(root, text=line, font=("Courier", 10)).pack()
        tk.Label(root, text="-"*80).pack()  # Horizontal line separator

    tk.Label(root, text="VANET Dataset Preview").pack()

    # Add horizontal lines between rows in VANET dataset preview
    vanet_lines = vanet_preview.split('\n')
    for line in vanet_lines:
        tk.Label(root, text=line, font=("Courier", 10)).pack()
        tk.Label(root, text="-"*80).pack()  # Horizontal line separator

    # Button to continue
    tk.Button(root, text="Continue", command=on_button_click).pack()

    root.mainloop()


# File paths
veremi_path = "C:\\Users\\ahmet\\OneDrive\\Masaüstü\\Sunum\\Yeni klasör\\Kod\\VeReMi-Dataset\\MixAll_0024\\VeReMi_7200_10800_2022-9-11_12_51_1"
vanet_file_path = "C:\\Users\\ahmet\\OneDrive\\Masaüstü\\Sunum\\Yeni klasör\\Kod\\datasets-for-VANET-master\\64\\64V_10PPS_Processed_C63_disc.csv"

# Load datasets
try:
    veremi_data = load_veremi_dataset(veremi_path)
    vanet_data = load_vanet_dataset(vanet_file_path)

    if veremi_data.empty:
        raise ValueError("VeReMi dataset is empty.")
    if vanet_data.empty:
        raise ValueError("VANET dataset is empty.")

    # Combine datasets
    data = pd.concat([veremi_data, vanet_data], axis=0, ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
except Exception as e:
    print("Error loading datasets:", str(e))
    exit()

# Show first 10 rows
show_first_10_lines(veremi_data, vanet_data)

# Feature engineering
data['packet_delay_variation'] = np.abs(data['rcvTime'] - data['rcvTime'].shift(1))
data['signal_to_noise_ratio'] = data['speed_x'] / (data['acl_x'] + 1e-6)

# Select features
features = ['rcvTime', 'position_x', 'position_y', 'position_z', 'speed_x', 'speed_y', 'speed_z',
            'acl_x', 'acl_y', 'acl_z', 'heading_x', 'heading_y', 'heading_z', 'packet_delay_variation',
            'signal_to_noise_ratio']
X = data[features]
y = data['label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)

# Cross-validation
cv_scores = cross_val_score(best_model, X_scaled, y, cv=10, scoring='accuracy')
print("\nCross-Validation Accuracy (10-fold):", np.mean(cv_scores))

# Test set evaluation
y_pred = best_model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Apply SMOTE to oversample the attacker class (class 1)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)

# Cross-validation
cv_scores = cross_val_score(best_model, X_imputed, y, cv=10, scoring='accuracy')
print("\nCross-Validation Accuracy (10-fold):", np.mean(cv_scores))

# Test set evaluation
y_pred = best_model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
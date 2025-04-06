import os
import json
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Paths
dataset_dir = "dataset"
model_path = "models/pose_model.pkl"
tolerances_path = "models/pose_tolerances.json"

# Angles to track
ANGLE_NAMES = ["elbow_angle", "knee_angle", "shoulder_tilt", "head_tilt", "hip_angle"]

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load Data
shot_data = defaultdict(lambda: defaultdict(list))  # {shot_type: {frame_pos: [angles]}}
X, y, video_labels = [], [], []

for filename in os.listdir(dataset_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(dataset_dir, filename)

        try:
            with open(filepath, "r") as f:
                keyframes = json.load(f)

            if not keyframes:
                print(f"Skipping {filename} (Empty JSON)")
                continue

            # Extract shot type from filename (everything before "_")
            shot_type = filename.split("_")[0]

            # Process each frame
            frame_angles = []
            for frame in keyframes:
                frame_pos = frame["frame"]  # Get frame position

                if "angles" not in frame:
                    print(f"Skipping frame {frame_pos} in {filename} (Missing 'angles' key)")
                    continue

                try:
                    angles = [frame["angles"][angle] for angle in ANGLE_NAMES]
                    frame_angles.append(angles)

                    # Store frame data in shot_data per frame position
                    shot_data[shot_type][frame_pos].append(angles)

                except KeyError as e:
                    print(f"Skipping frame {frame_pos} in {filename} (Missing angle: {e})")
                    continue

            if not frame_angles:
                print(f"Skipping {filename} (No valid frames)")
                continue

            # Use mean values of angles across frames as a feature vector for the video
            mean_angles = np.mean(frame_angles, axis=0)

            X.append(mean_angles)
            y.append(shot_type)
            video_labels.append(filename)

        except json.JSONDecodeError:
            print(f"Skipping {filename} (Corrupt JSON)")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Check if data is sufficient
if len(X) == 0:
    raise ValueError("No valid data found. Check dataset folder.")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Model
model = RandomForestClassifier(n_estimators=100, random_state=45)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)

# Compute accuracy per shot type
shot_accuracies = defaultdict(lambda: {"correct": 0, "total": 0})

for actual, predicted in zip(y_test, y_pred):
    shot_accuracies[actual]["total"] += 1
    if actual == predicted:
        shot_accuracies[actual]["correct"] += 1

# Print overall accuracy
print(f"Overall Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Print per-shot accuracy
print("\nShot-wise Accuracy:")
for shot, counts in shot_accuracies.items():
    accuracy = (counts["correct"] / counts["total"]) * 100
    print(f"{shot}: {accuracy:.2f}% ({counts['correct']} / {counts['total']})")

# Save Model
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Save Pose Tolerances (Per Shot Type & Frame Position)
pose_tolerances = {}

for shot_type, frames_data in shot_data.items():
    pose_tolerances[shot_type] = {}

    for frame_pos, frame_values in frames_data.items():
        shot_array = np.array(frame_values)  # Convert list to NumPy array

        if len(shot_array) > 1:
            weights = np.linspace(0.5, 2, num=len(shot_array))  # Weight later frames more

            pose_tolerances[shot_type][frame_pos] = {
                name: {
                    "mean": float(np.average(shot_array[:, i], weights=weights)),
                    "std": float(np.sqrt(np.average((shot_array[:, i] - np.average(shot_array[:, i], weights=weights))**2, weights=weights))) * (2/3)
                }
                for i, name in enumerate(ANGLE_NAMES)
            }

# Save updated tolerances
with open(tolerances_path, "w") as f:
    json.dump(pose_tolerances, f, indent=4)

print(f"Pose tolerances saved to {tolerances_path}")

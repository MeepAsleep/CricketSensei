import json
import os
import joblib
import sys
from collections import Counter
from analyzer import extract_keyframes

# Paths for models and keyframes
POSE_MODEL_PATH = "../code/models/pose_model.pkl"
TOLERANCE_PATH = "../code/models/pose_tolerances.json"
USER_KEYFRAME_PATH = "../code/user_data/user_keyframes.json"
OUTPUT_JSON_PATH = "comparison_output.json"

# Load JSON helper
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Save JSON helper
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Load trained model & tolerances
def load_models():
    ml_model = joblib.load(POSE_MODEL_PATH)
    tolerances = load_json(TOLERANCE_PATH)
    return ml_model, tolerances

# Get feedback based on deviation (with halved std)
def get_quality_and_feedback(param, deviation, user_value, ref_range):
    direction = "Increase" if user_value < ref_range["mean"] else "Decrease"
    threshold = ref_range["std"] * 0.5  # Halved std for tighter tolerances

    if deviation <= threshold:
        return "Perfect", None
    elif deviation <= threshold * 2:
        return "Great", None
    elif deviation <= threshold * 3:
        return "Good", f"{direction} your {param.replace('_', ' ')} for better results."
    else:
        feedback = {
            "elbow_angle": f"{direction} your elbow angle for better alignment.",
            "knee_angle": f"{direction} your knee angle to match the reference posture.",
            "shoulder_tilt": f"{direction} your shoulder tilt for better posture.",
            "head_tilt": f"{direction} your head tilt to improve alignment.",
            "hip_angle": f"{direction} your hip angle for better posture."
        }.get(param, f"{direction} your {param.replace('_', ' ')} for better posture.")
        return "Needs Improvement", feedback

# Generate and print phase-wise feedback
def print_phase_feedback(frame_feedback):
    phase_ranges = {
        "Starting Phase": range(0, 3),
        "Middle Phase": range(3, 7),
        "Ending Phase": range(7, 10)
    }

    print("\nGeneralized Feedback:")

    for phase, frame_range in phase_ranges.items():
        phase_issues = []

        for frame in frame_feedback:
            if frame["frame"] - 1 in frame_range:
                phase_issues.extend(frame["issues"])

        issue_counts = Counter(phase_issues)
        filtered_issues = [issue for issue, count in issue_counts.items() if count >= 2]

        if filtered_issues:
            print(f"\n{phase}:")
            for issue in filtered_issues:
                print(f"  - {issue}")

# Compare user keyframes against reference tolerances
def compare_keyframes(ml_model, tolerances, user_keyframes, actual_shot_type):
    if actual_shot_type not in tolerances:
        sys.exit(1)

    shot_tolerances = tolerances[actual_shot_type]
    frame_feedback = []

    total_score = 0
    max_score = 0

    for frame_idx, user_frame in enumerate(user_keyframes):
        if str(frame_idx + 1) not in shot_tolerances:
            continue

        frame_result = {"frame": frame_idx + 1, "analysis": {}, "issues": []}
        frame_tolerances = shot_tolerances[str(frame_idx + 1)]
        features = []

        frame_score = 0
        angle_count = 0

        for param, user_value in user_frame["angles"].items():
            if param in frame_tolerances:
                ref_range = frame_tolerances[param]
                deviation = abs(user_value - ref_range["mean"])
                quality, advice = get_quality_and_feedback(param, deviation, user_value, ref_range)

                # Score mapping
                score_map = {
                    "Perfect": 1.0,
                    "Great": 0.8,
                    "Good": 0.5,
                    "Needs Improvement": 0.2
                }
                frame_score += score_map[quality]
                angle_count += 1

                frame_result["analysis"][param] = quality
                if advice:
                    frame_result["issues"].append(advice)

                features.append(user_value)

        if angle_count > 0:
            total_score += frame_score
            max_score += angle_count

        prediction = ml_model.predict([features])[0]
        frame_result["prediction"] = "Correct Form" if prediction == 1 else "Incorrect Form"

        frame_feedback.append(frame_result)

    accuracy = (total_score / max_score * 100) if max_score > 0 else 0.0
    return frame_feedback, round(accuracy, 2)


# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python compare.py <user_video_path> <shot_type>")

    user_video_path = sys.argv[1]
    actual_shot_type = sys.argv[2]

    # Step 1: Extract keyframes
    keyframes = extract_keyframes(user_video_path)
    os.makedirs(os.path.dirname(USER_KEYFRAME_PATH), exist_ok=True)
    save_json(keyframes, USER_KEYFRAME_PATH)

    # Step 2: Load models & keyframes
    ml_model, tolerances = load_models()
    user_keyframes = load_json(USER_KEYFRAME_PATH)

    # Step 3: Compare keyframes
    frame_feedback, shot_accuracy = compare_keyframes(ml_model, tolerances, user_keyframes, actual_shot_type)

    # Step 4: Predict shot type
    predicted_shot = None
    if user_keyframes:
        all_angles = []
        for frame in user_keyframes:
            angles = [frame["angles"].get(param, 0) for param in ["elbow_angle", "knee_angle", "shoulder_tilt", "head_tilt", "hip_angle"]]
            all_angles.append(angles)

        mean_angles = [sum(col) / len(col) for col in zip(*all_angles)]
        predicted_shot = ml_model.predict([mean_angles])[0]
        print(f"\n🔍 Predicted Shot Type: {predicted_shot}")

    # Step 5: Print phase-wise feedback
    print_phase_feedback(frame_feedback)

    # Step 6: Save full feedback to JSON
    final_output = {
        "frames": frame_feedback,
        "summary": {
            "predicted_shot_type": predicted_shot,
            "actual_shot_type": actual_shot_type,
            "shot_accuracy_percent": round(shot_accuracy, 2)
        }
    }

    save_json(final_output, OUTPUT_JSON_PATH)
    print(f"\nComparison saved to {OUTPUT_JSON_PATH}")

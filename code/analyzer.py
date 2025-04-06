import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Paths
shot_data_dir = "shot_data"
dataset_dir = "dataset"
output_frames_dir = "../UI/static/output_frames"

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
os.makedirs(output_frames_dir, exist_ok=True)

# Ensure dataset directory exists
os.makedirs(dataset_dir, exist_ok=True)

def extract_keyframes(video_path, flag=1):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    frame_id = 0
    selected_frames = {2, 6, 9}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = results.pose_landmarks.landmark
            angles = calculate_angles(keypoints)

            keyframes.append({
                "frame": frame_id,
                "angles": angles
            })

        if flag == 1 and frame_id in selected_frames:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            # Save the annotated frame
            frame_path = os.path.join(output_frames_dir, f"{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)

    cap.release()
    return keyframes

def calculate_angles(keypoints):
    def get_angle(p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        return round(angle, 2)

    return {
        "elbow_angle": get_angle(keypoints[11], keypoints[13], keypoints[15]),    # Left elbow
        "knee_angle": get_angle(keypoints[23], keypoints[25], keypoints[27]),     # Left knee
        "shoulder_tilt": get_angle(keypoints[11], keypoints[23], keypoints[24]),  # Shoulder tilt
        "hip_angle": get_angle(keypoints[23], keypoints[25], keypoints[27]),      # Hip angle
        "head_tilt": get_angle(keypoints[11], keypoints[0], keypoints[12])        # Head tilt (left shoulder, nose, right shoulder)
    }

def main():
    for video_file in os.listdir(shot_data_dir):
        if video_file.endswith(".avi"):
            video_path = os.path.join(shot_data_dir, video_file)

            # Extract keyframes
            keyframes = extract_keyframes(video_path, 0)

            # Construct the output filename correctly
            output_file = os.path.join(dataset_dir, f"{video_file.replace('.avi', '.json')}")

            # Save the keyframes as JSON
            with open(output_file, "w") as f:
                json.dump(keyframes, f, indent=4)

            print(f"Keyframes extracted: {output_file}")

if __name__ == "__main__":
    main()

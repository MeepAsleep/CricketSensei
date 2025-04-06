import cv2, os
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

dataset_path = "flick"

def is_left_handed(video):
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Skipping {video} (Unreadable)")
        return False

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        lw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        rh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        return lw.x > rh.x  # Left wrist is to the right of right hip → Left-hander
    return False

for vid in os.listdir(dataset_path):
    path = os.path.join(dataset_path, vid)
    if vid.endswith(".avi"):  # Ensure only .avi files are processed
        if is_left_handed(path):
            os.remove(path)
            print(f"Deleted: {vid}")

print("Left-handed videos removed!")

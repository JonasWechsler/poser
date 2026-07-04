import cv2
import mediapipe as mp
import numpy as np

# =====================================================================
# PURE OPENCV VISUALIZATION HELPER (No mediapipe.framework needed!)
# =====================================================================
def draw_landmarks_on_image(rgb_image, detection_result):
    if not detection_result.pose_landmarks:
        return rgb_image

    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    # MediaPipe Pose connection pairs (mapping which landmark connects to which)
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Shoulders and arms
        (11, 23), (12, 24), (23, 24),                     # Torso
        (23, 25), (24, 26), (25, 27), (26, 28),           # Legs
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32) # Feet
    ]

    for pose_landmarks in detection_result.pose_landmarks:
        # 1. Convert normalized coordinates to pixel coordinates
        points = {}
        for idx, landmark in enumerate(pose_landmarks):
            # Only draw confident landmarks
            if landmark.visibility > 0.5:
                px = int(landmark.x * width)
                py = int(landmark.y * height)
                points[idx] = (px, py)

        # 2. Draw connections (lines)
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in points and end_idx in points:
                cv2.line(annotated_image, points[start_idx], points[end_idx], (0, 255, 0), 2)

        # 3. Draw keypoints (circles)
        for idx, pt in points.items():
            cv2.circle(annotated_image, pt, 5, (0, 0, 255), -1)

    return annotated_image

# =====================================================================
# MAIN PIPELINE
# =====================================================================

# STEP 1: Import the necessary modules.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create a PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.jpg")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result and visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# Convert back to BGR because MediaPipe works in RGB, but OpenCV handles BGR locally
bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

# Local Window Display logic
cv2.imshow("MediaPipe Pose Detection", bgr_annotated_image)

print("Displaying image window. Press any key on your keyboard to close it...")
cv2.waitKey(0)  # Keeps the window open until you press a key
cv2.destroyAllWindows()

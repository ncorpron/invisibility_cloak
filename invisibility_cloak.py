import cv2
import numpy as np
import os
import subprocess

# --- Paths ---
project_folder = r"C:\Users\ncorp\PycharmProjects\PythonProject_invisibility_cloak"
video_path = os.path.join(project_folder, "IMG_4408.MOV")
background_path = os.path.join(project_folder, "IMG_4409.JPG")
final_output_path = os.path.join(project_folder, "invisibility_cloak_final_h264.mp4")
ffmpeg_path = r"C:\ffmpeg_temp\ffmpeg-2025-09-04-git-2611874a50-full_build\bin\ffmpeg.exe"

# --- Load background ---
background = cv2.imread(background_path)
if background is None:
    raise FileNotFoundError(f"Background image not found at: {background_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Video file not found at: {video_path}")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Resize background
background = cv2.resize(background, (frame_width, frame_height))

# Temporary raw output path
raw_temp_path = os.path.join(project_folder, "temp_invisibility_raw.mp4")
out = cv2.VideoWriter(raw_temp_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

# HSV range for turquoise blanket
lower_turquoise = np.array([75, 30, 30])
upper_turquoise = np.array([105, 255, 255])

kernel = np.ones((5, 5), np.uint8)

# --- Trackbar callback (does nothing, required by OpenCV) ---
def nothing(x):
    pass

# --- Create window and sliders ---
cv2.namedWindow("Invisibility Cloak")
cv2.createTrackbar("X Shift", "Invisibility Cloak", 0, frame_width, nothing)
cv2.createTrackbar("Y Shift", "Invisibility Cloak", 0, frame_height, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get slider values
    x_shift = cv2.getTrackbarPos("X Shift", "Invisibility Cloak")
    y_shift = cv2.getTrackbarPos("Y Shift", "Invisibility Cloak")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_turquoise, upper_turquoise)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_inv = cv2.bitwise_not(mask)

    # Shift the background
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted_background = cv2.warpAffine(background, M, (frame_width, frame_height))

    res1 = cv2.bitwise_and(shifted_background, shifted_background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    final_frame = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Show video in PyCharm
    cv2.imshow("Invisibility Cloak", final_frame)
    out.write(final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# --- Re-encode with H.264 for LinkedIn/GitHub ---
ffmpeg_command = [
    ffmpeg_path,
    '-i', raw_temp_path,
    '-c:v', 'libx264',
    '-preset', 'fast',
    '-crf', '22',
    '-pix_fmt', 'yuv420p',
    final_output_path
]

subprocess.run(ffmpeg_command, check=True)
os.remove(raw_temp_path)
print(f"Final H.264 video ready: {final_output_path}")

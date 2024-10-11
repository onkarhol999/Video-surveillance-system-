import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('violence_detection_model.h5')

# Function to extract frames from a video and predict if they are violent or not
def predict_violence_in_video(video_path, model, img_size=(128, 128)):
    # Capture the video
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Get frames per second

    frame_number = 0
    success, frame = video_capture.read()
    violent_frames = 0
    total_frames = 0

    while success:
        # Calculate the current timestamp in seconds
        current_time_sec = frame_number / fps

        # Check if the current frame is approximately at a second boundary
        if int(current_time_sec) == current_time_sec:  # Process every full second frame
            # Preprocess the frame for the model
            frame_resized = cv2.resize(frame, img_size)
            frame_normalized = frame_resized / 255.0
            frame_array = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

            # Predict the label (0 = non-violent, 1 = violent)
            prediction = model.predict(frame_array)
            predicted_label = 1 if prediction[0] > 0.5 else 0

            # Count frames based on prediction
            if predicted_label == 1:
                violent_frames += 1
            total_frames += 1

            print(f"Frame {frame_number}: {'Violent' if predicted_label == 1 else 'Non-Violent'}")

        # Read the next frame
        success, frame = video_capture.read()
        frame_number += 1

    video_capture.release()

    # Decision based on the proportion of violent frames
    violence_ratio = violent_frames / total_frames if total_frames > 0 else 0
    print(f"Violent frames: {violent_frames} / {total_frames} ({violence_ratio * 100:.2f}%)")

    # You can set a threshold for declaring the video as violent
    if violence_ratio > 0.5:  # For example, if more than 20% of frames are violent
        print("Violence detected in the video.")
        return True
    else:
        print("No significant violence detected in the video.")
        return False

# Usage example:
video_path = './faight.mp4'  # Path to the video you want to test

# Call the function to predict violence in the video
is_violent = predict_violence_in_video(video_path, model)

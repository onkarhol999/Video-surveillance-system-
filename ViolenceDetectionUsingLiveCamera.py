import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model('violence_detection_model.h5')

# Function to predict violence in live camera feed
def predict_violence_in_live_camera(model, img_size=(128, 128), interval=10):
    # Open the laptop camera (camera index 0 is usually the default camera)
    video_capture = cv2.VideoCapture(0)
    
    violent_frames = 0
    total_frames = 0
    start_time = time.time()  # Record the start time

    while True:
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Capture a frame once every second
        if int(elapsed_time) % 1 == 0:
            # Capture a frame from the camera
            success, frame = video_capture.read()
            if not success:
                break

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

            # Show prediction on the frame
            label_text = "Violent" if predicted_label == 1 else "Non-Violent"
            color = (0, 0, 255) if predicted_label == 1 else (0, 255, 0)
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            print(f"Frame {total_frames}: {label_text}")

            # Display the frame in a window
            cv2.imshow("Live Violence Detection", frame)

        # Provide feedback every 10 seconds
        if elapsed_time >= interval:
            # Calculate the violence ratio for the 10-second interval
            violence_ratio = violent_frames / total_frames if total_frames > 0 else 0
            print(f"\nViolent frames: {violent_frames} / {total_frames} ({violence_ratio * 100:.2f}%)")

            # Alert based on the threshold
            if violence_ratio > 0.5:
                alert_text = "ALERT: Violence Detected!"
                print(alert_text)
            else:
                print("No significant violence detected in the last 10 seconds.")

            # Reset counts and start time for the next interval
            violent_frames = 0
            total_frames = 0
            start_time = time.time()  # Reset the start time

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the live camera violence detection
predict_violence_in_live_camera(model)

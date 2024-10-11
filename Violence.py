import os
import cv2
import csv

def extract_frames_and_label(folder_path, output_csv, frames_output_folder):
    # List to store the frame details (video name, frame number, label)
    data = []

    # Ensure the output folder for frames exists
    if not os.path.exists(frames_output_folder):
        os.makedirs(frames_output_folder)

    # Iterate over all files in the folder
    for video_file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_file)

        # Check if it's a video file
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Capture the video
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)  # Get the frames per second

            frame_number = 0
            success, frame = video_capture.read()

            while success:
                # Calculate the current timestamp in seconds
                current_time_sec = frame_number / fps

                # Check if the current frame is approximately at a second boundary
                if int(current_time_sec) == current_time_sec:  # Save every full second frame
                    frame_name = f"{os.path.splitext(video_file)[0]}_frame_{frame_number}.jpg"
                    frame_path = os.path.join(frames_output_folder, frame_name)
                    
                    # Save the frame as an image
                    cv2.imwrite(frame_path, frame)

                    # Add the frame info to the CSV data
                    data.append([frame_name, 1])

                # Read the next frame
                success, frame = video_capture.read()
                frame_number += 1

            video_capture.release()

    # Write all the frames and their labels to the CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame_name', 'label'])  # Write the header
        writer.writerows(data)  # Write the data rows

    print(f"Frames have been saved to {frames_output_folder} and labels have been written to {output_csv}")

# Usage example:
folder_path = "./Dataset/ViolenceVideo"  # Folder where your videos are stored
output_csv = 'Violence_video_frames_labels.csv'  # Output CSV file
frames_output_folder = './Frames/ViolenceVideoFrames'  # Folder where frames will be saved

# Call the function
extract_frames_and_label(folder_path, output_csv, frames_output_folder)

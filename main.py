import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load the CSV files containing metadata
violence_csv = 'Violence_video_frames_labels.csv'
normal_csv = 'Normal_video_frames_labels.csv'

# Folders where frames are stored
violence_frames_folder = './Frames/ViolenceVideoFrames'
normal_frames_folder = './Frames/NormalVideoFrames'

# Read the CSV files
violence_df = pd.read_csv(violence_csv)
normal_df = pd.read_csv(normal_csv)

# Add the full path to the frames in the DataFrame
violence_df['frame_path'] = violence_df['frame_name'].apply(lambda x: os.path.join(violence_frames_folder, x))
normal_df['frame_path'] = normal_df['frame_name'].apply(lambda x: os.path.join(normal_frames_folder, x))

# Combine both DataFrames
df = pd.concat([violence_df, normal_df])

# Shuffle the data to randomize the order
df = shuffle(df)

# Function to load and preprocess the images
def load_images(df, img_size=(128, 128)):
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = row['frame_path']
        label = row['label']
        
        # Read the image
        image = cv2.imread(img_path)
        if image is not None:
            # Resize the image to the target size
            image = cv2.resize(image, img_size)
            # Normalize the pixel values (0 to 1)
            image = image / 255.0
            images.append(image)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Load the image data and labels
X, y = load_images(df, img_size=(128, 128))

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
def create_model(input_shape):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer (Binary classification: Normal or Violence)
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Create the CNN model
input_shape = (128, 128, 3)  # Shape of input images (128x128 with 3 color channels)
model = create_model(input_shape)

# Display the model architecture
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# # Save the trained model
# model.save('violence_detection_model.h5')

# # Plot training history (optional)
# import matplotlib.pyplot as plt

# def plot_training_history(history):
#     plt.figure(figsize=(12, 4))

#     # Accuracy plot
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Accuracy over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     # Loss plot
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Loss over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.show()

# # Call the plotting function
# plot_training_history(history)

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

# Function to extract frames from videos
def extract_frames(video_path, output_folder, frame_rate=1): # frame_rate is the number of frames to skip
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0: # Save the frame every `frame_rate` frames
            frame_filename = os.path.join(output_folder, f"frame_{count}.jpg") # Path to the frame file
            cv2.imwrite(frame_filename, frame)
        count += 1
    cap.release()

# Extract frames from all videos in the dataset
def process_videos(input_folder, output_folder, frame_rate=1):
    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp4"):
                print ("extracting frames from", file)
                video_path = os.path.join(subdir, file) # Path to the video file
                class_folder = os.path.basename(subdir) # Class folder (real or fake)
                output_class_folder = os.path.join(output_folder, class_folder) #
                extract_frames(video_path, output_class_folder, frame_rate)

# Paths to the video dataset
train_videos_path = 'UADFV/train'
validation_videos_path = 'UADFV/validation'
test_videos_path = 'UADFV/test'

# Paths to the extracted frames
train_frames_path = 'UADFV_frames/train'
validation_frames_path = 'UADFV_frames/validation'
test_frames_path = 'UADFV_frames/test'

# Extract frames from the videos
process_videos(train_videos_path, train_frames_path)
process_videos(validation_videos_path, validation_frames_path)
process_videos(test_videos_path, test_frames_path)

# Load and preprocess the dataset
datagen = ImageDataGenerator(rescale=1.0/255.0) # Image data generator for loading and augmenting images
# 1.0/255.0 is used to scale the pixel values to the range [0, 1]

train_generator = datagen.flow_from_directory(
    train_frames_path,
    target_size=(299, 299), # Resize the images to 256x256
    batch_size=32,
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    validation_frames_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

test_generator = datagen.flow_from_directory(
    test_frames_path,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

# Define the XceptionNet model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')
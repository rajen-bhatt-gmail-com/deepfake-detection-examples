import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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
    target_size=(256, 256), # Resize the images to 256x256
    batch_size=32,
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    validation_frames_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

test_generator = datagen.flow_from_directory(
    test_frames_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

# Define the MesoNet model
def create_meso_net(input_shape=(256, 256, 3)):
    model = Sequential() # Create a sequential model, meaning that the layers are added in sequence

    # First block
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    # Conv2D layer with 8 filters, kernel size of 3x3, padding of 'same', and ReLU activation function
    # 256x256 channel when convolved with 3x3 kernel and padding of 'same' will result in 256x256 channel
    # Each filter in the conv2D layer has a size of 3x3, however, since the input has 3 channels, the actual kernel size is 3x3x3
    # Each 3x3x3 filter slides over the input image. At each position, the filter performs element-wise multiplication with the 
    # corresponding 3x3x3 patch of the input image. The results of these multiplications are summed up to produce a single value.
    # This single value becomes a part of the output feature map.
    # Each channel is convolved with 8 filters, resulting in 256x256x8 channels
    # ReLU activation function is applied to the output of the conv2D layer, f(x) = max(0, x), this means that all negative values 
    # are set to zero, otherwise it outputs the input value 
    
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    # MaxPooling2D layer with a pool size of 2x2 and padding of 'same'
    # The 2x2 pool size means that the maximum value in each 2x2 region of the input is selected to be a part of the output feature map
    # The padding of 'same' means that the input is zero-padded so that the output has the same height and width as the input
    # Therefore, the output of this layer will be 256x256x8 channelsd

    # Second block
    model.add(Conv2D(8, (5, 5), padding='same', activation='relu')) # 256x256x8 channels
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same')) # 256x256x8 channels

    # Third block
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu')) # 256x256x16 channels
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same')) # 256x256x16 channels

    # Fourth block
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu')) # 256x256x16 channels
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same')) # 256x256x16 channels

    # Fully connected layers
    model.add(Flatten()) # Flatten the output of the previous layer to a 1D array, 256x256x16 = 1048576 = 2^20
    model.add(Dropout(0.5)) # Dropout layer with a rate of 0.5, this means that 50% of the neurons will be randomly set to zero during 
    # training, meaning 50% of 1048576 neurons will be set to zero
    model.add(Dense(16, activation='relu')) # Dense layer with 16 neurons and ReLU activation function, 16x1 weights and 1 bias
    # Number of parameters = 1048576 * 16 + 16 = 16777232, +16 is for the bias
    model.add(Dropout(0.5)) # Dropout layer with a rate of 0.5
    model.add(Dense(1, activation='sigmoid'))

    return model

# Create the MesoNet model
meso_net = create_meso_net()
meso_net.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
meso_net.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20
)

# Evaluate the model
loss, accuracy = meso_net.evaluate(test_generator)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
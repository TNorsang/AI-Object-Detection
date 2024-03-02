from pytube import YouTube
import cv2  
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import random
import datetime
from keras import layers
from keras.preprocessing.image import img_to_array, array_to_img
import keras


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
 
def preprocess_detected_objects(image, predictions, resize_dim, output_folder, frame_id, video_folder_name):
    # Create a specific folder for the current video inside the output_folder
    video_specific_folder = os.path.join(output_folder, video_folder_name)
    ensure_dir(video_specific_folder)
    
    objects_detected_folder = os.path.join(video_specific_folder, 'ObjectsDetected')
    ensure_dir(objects_detected_folder)
    
    for i, box in enumerate(predictions['boxes']):
        box = [int(coord) for coord in box]  # Convert coordinates to integers
        cropped_img = image.crop(box)  # Crop the detected object
        resized_img = cropped_img.resize(resize_dim)  # Resize to the input shape of the autoencoder
        
        # Construct a unique filename for each processed image
        image_filename = f"object_{frame_id}_{i}.png"
        image_path = os.path.join(video_specific_folder, image_filename)
        
        # Save the preprocessed image
        resized_img.save(image_path)
 
def build_autoencoder(input_shape=(64, 64, 3)):
    input_img = keras.Input(shape=input_shape)
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder
                
def extract_frames_and_detect_objects(video_file_path, model, vidId, interval, output_folder, video_folder_name):
    vidcap = cv2.VideoCapture(video_file_path)
    success, image = vidcap.read()
    count = 0
    saved_frames = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames_dir = os.path.join(output_folder, f"{video_folder_name}")
    ensure_dir(frames_dir)
    results = []

    while success and saved_frames < 15:
        if count % (int(fps) * interval) == 0:
            frame_path = os.path.join(frames_dir, f"frame{saved_frames}.jpg")
            cv2.imwrite(frame_path, image)
            prediction = detect_objects(model, frame_path)
            if 'labels' in prediction:
                draw_bounding_boxes(frame_path, prediction)
                preprocess_detected_objects(Image.fromarray(image), prediction, resize_dim=(64, 64), output_folder=output_folder, frame_id=saved_frames, video_folder_name=video_folder_name)
                for i, label in enumerate(prediction['labels']):
                    box_info = prediction['boxes'][i]
                    if torch.is_tensor(box_info):
                        box_info = box_info.tolist()
                    results.append([
                        vidId,
                        saved_frames,
                        datetime.timedelta(seconds=int(count / fps)),
                        i,  # Detected object ID
                        label,
                        prediction['scores'][i],
                        box_info
                    ])
                saved_frames += 1
        success, image = vidcap.read()
        count += 1

    columns = ['vidId', 'frameNum', 'timestamp(H:MM:SS)', 'detectedObjId', 'detectedObjClass', 'confidence', 'bbox info']
    return pd.DataFrame(results, columns=columns)

def encode_objects_with_autoencoder(input_folder, output_folder, autoencoder_model):
    # Iterate through each image file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):  # Assuming all objects are saved as PNG files
            # Load the image
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if necessary
            
            # Convert the image to array and normalize
            img_array = img.astype('float32') / 255.0
            
            # Encode the image using the autoencoder
            encoded_representation = autoencoder_model.predict(np.expand_dims(img_array, axis=0))
            
            # Save the encoded representation
            output_filename = os.path.join(output_folder, filename.replace(".png", "_encoded.npy"))
            np.save(output_filename, encoded_representation)
            
            print(f"Encoded representation saved to: {output_filename}")

preprocessed_folder = "//Users/norsangnyandak/Documents/Spring 2024/CS370-102 Introduction to Artificial Intelligence/AI-Object-Detection/Videos/Preprocessed"
ensure_dir(preprocessed_folder)

encoded_output_folder = "//Users/norsangnyandak/Documents/Spring 2024/CS370-102 Introduction to Artificial Intelligence/AI-Object-Detection/Videos/Encoded"

        
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


coco_class = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

def safe_filename(filename):
    keepcharacters = (' ', '.', '_', '-')
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()

def download_video(video_url, output_path):
    try:
        yt = YouTube(video_url)
        yt.bypass_age_gate()
        caption = yt.captions["a.en"]
        if caption == {}:
            caption = yt.captions["en"]
            
        stream = yt.streams.get_highest_resolution()
        safe_title = safe_filename(stream.title)
        caption_title = safe_filename(stream.title + " Caption")
        video_folder = os.path.join(output_path, safe_title)
        full_path = os.path.join(video_folder, safe_title + ".mp4")
        caption_path = os.path.join(video_folder, caption_title + ".srt")  # Path for the SRT file

        # Make sure the directory exists
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        
        # Download the video
        stream.download(video_folder, safe_title + ".mp4")
        
        # Write the XML captions to an SRT file
        if caption:
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption.xml_captions)
            print(f"Caption downloaded successfully to: {caption_path}")
        else:
            print("No caption was found.")
            
        print(f"Video downloaded successfully to: {full_path}")
        return video_folder, full_path  # Return the folder path where the video is saved
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None

def load_model():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    return model

def detect_objects(model, frame_path, confidence_threshold=0.7):
    img = Image.open(frame_path).convert("RGB")
    img_tensor = F.to_tensor(img)
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    pred_scores = prediction['scores']
    high_conf_pred_indices = [i for i, score in enumerate(pred_scores) if score > confidence_threshold]
    high_conf_predictions = {
        # Use get() method to safely retrieve the class name using class ID as key; default to 'unknown' if key not found
        'labels': [coco_class.get(i, 'unknown') for i in prediction['labels'][high_conf_pred_indices].tolist()],
        'scores': prediction['scores'][high_conf_pred_indices].tolist(),
        'boxes': prediction['boxes'][high_conf_pred_indices].tolist()
    }

    return high_conf_predictions

def draw_bounding_boxes(frame_path, predictions):
    image = cv2.imread(frame_path)
    if 'boxes' in predictions:
        for i, box in enumerate(predictions['boxes']):
            box = [int(coord) for coord in box]  
            start_point = tuple(box[:2])
            end_point = tuple(box[2:])
            color = (0, 0, 255) 
            thickness = 2
            cv2.rectangle(image, start_point, end_point, color, thickness)
            label = f"{predictions['labels'][i]}: {predictions['scores'][i]:.2f}"
            cv2.putText(image, label, (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(frame_path.replace('.jpg', '_detected.jpg'), image)

def extract_frames_and_detect_objects(video_file_path, model, vidId, interval=1):
    vidcap = cv2.VideoCapture(video_file_path)
    video_path = os.path.join(video_folder, os.listdir(video_folder)[0])
    success, image = vidcap.read()
    count = 0
    saved_frames = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames_dir = os.path.join(video_folder, "frames")
    results = []

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    while success and saved_frames < 15:
        if count % (int(fps) * interval) == 0:
            timestamp = count / fps  
            # Split into integer and fractional parts
            integer_part = int(timestamp)
            fractional_part = timestamp - integer_part
            # Use datetime.timedelta for the integer part and format fractional part to two digits
            formatted_time = str(datetime.timedelta(seconds=integer_part))
            formatted_fractional = f"{fractional_part:.2f}"[1:]  # Extracting fractional part
            final_timestamp = formatted_time + formatted_fractional

            frame_path = os.path.join(frames_dir, f"frame{saved_frames}.jpg")
            cv2.imwrite(frame_path, image)
            prediction = detect_objects(model, frame_path)
            
            if 'labels' in prediction:
                draw_bounding_boxes(frame_path, prediction)
                for i, label in enumerate(prediction['labels']):
                    box_info = prediction['boxes'][i]
                    if torch.is_tensor(box_info):
                        box_info = box_info.tolist()
                    results.append([
                        vidId,
                        saved_frames,
                        final_timestamp,  # Use the formatted timestamp
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


model = load_model()

video_urls = [
    "https://www.youtube.com/watch?v=wbWRWeVe1XE",
    "https://www.youtube.com/watch?v=FlJoBhLnqko",
    "https://www.youtube.com/watch?v=Y-bVwPRy_no"
]

output_path = "/Users/norsangnyandak/Documents/Spring 2024/CS370-102 Introduction to Artificial Intelligence/AI-Object-Detection/Videos"

for url in video_urls:
    video_folder, video_file_path = download_video(url, output_path)  # Unpack the returned values
    if video_folder and video_file_path:  # Check if both values are returned successfully
        videoID = random.randrange(100)
        results_df = extract_frames_and_detect_objects(video_file_path, model, videoID, interval=1)  # Pass the direct video file path
        print(results_df)

    

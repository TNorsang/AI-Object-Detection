from pytube import YouTube
import cv2  
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def download_video(video_url, output_path):
    try:
        if os.path.exists(output_path):
            print(f"The video already exists at: {output_path}")
            return

        yt = YouTube(video_url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path)
        print(f"Video downloaded successfully to: {output_path}")
    except Exception as e:
        print(f"Error downloading video: {str(e)}")

video_url = "https://www.youtube.com/watch?v=wbWRWeVe1XE"
output_path = "/Users/norsangnyandak/Documents/Spring 2024/CS370-102 Introduction to Artificial Intelligence/AI-Object-Detection/Videos"
download_video(video_url, output_path)

def safe_filename(filename):
    """Generate a safe filename for most file systems."""
    keepcharacters = (' ', '.', '_', '-')
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()

def load_model():
    """Load a pre-trained Fast R-CNN model."""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects(model, frame_path, confidence_threshold=0.5):
    """Detect objects in an image frame with a confidence threshold."""
    img = Image.open(frame_path).convert("RGB")
    img_tensor = F.to_tensor(img)
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    pred_scores = prediction['scores']
    high_conf_pred_indices = [i for i, score in enumerate(pred_scores) if score > confidence_threshold]
    high_conf_predictions = {
        'labels': [COCO_INSTANCE_CATEGORY_NAMES[i] for i in prediction['labels'][high_conf_pred_indices].tolist()],
        'scores': prediction['scores'][high_conf_pred_indices].tolist(),
        'boxes': prediction['boxes'][high_conf_pred_indices].tolist()
    }

    return high_conf_predictions

def draw_bounding_boxes(frame_path, predictions):
    """Draw bounding boxes and labels on the image."""
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

def extract_frames_and_detect_objects(video_path, model, interval=1):
    """Extract frames from the video at the specified interval and detect objects."""
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    saved_frames = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames_dir = os.path.join(os.path.dirname(video_path), "frames")
    results = []

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    while success and saved_frames < 50:
        if count % (int(fps) * interval) == 0:
            # Detect objects in the saved frame
            frame_path = os.path.join(frames_dir, f"frame{saved_frames}.jpg")
            cv2.imwrite(frame_path, image)
            prediction = detect_objects(model, frame_path)
            
            # Check if any objects are detected before saving the frame
            if 'labels' in prediction:
                draw_bounding_boxes(frame_path, prediction)  # Draw bounding boxes on the image
                for i, label in enumerate(prediction['labels']):
                    results.append({
                        'frameNum': saved_frames,
                        'detectedObjClass': label,
                        'confidence': prediction['scores'][i],
                        'bbox info': prediction['boxes'][i]
                    })
                saved_frames += 1

        success, image = vidcap.read()
        count += 1

    return pd.DataFrame(results)

model = load_model()

VideoTitles = ["What Does High-Quality Preschool Look Like  NPR Ed.mp4"]
# Why Its Usually Hotter In A City  Lets Talk  NPR.mp4
# What Does High-Quality Preschool Look Like  NPR Ed.mp4
# How Green Roofs Can Help Cities  NPR.mp4

for title in VideoTitles:
    video_path = f"/Users/norsangnyandak/Documents/Spring 2024/CS370-102 Introduction to Artificial Intelligence/AI-Object-Detection/Videos/{title}"
    results_df = extract_frames_and_detect_objects(video_path, model, interval=1)
    print(results_df)
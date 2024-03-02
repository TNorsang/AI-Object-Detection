import torch
import torchvision.transforms as transforms
from torchvision.models.detection import retinanet_resnet50_fpn
import numpy as np
import cv2
import random
from pycocotools.coco import COCO

# Load the pre-trained RetinaNet model
model = retinanet_resnet50_fpn(
    pretrained=True, weights="RetinaNet_ResNet50_FPN_Weights.DEFAULT")

# Set the model to evaluation mode
model.eval()

# Load COCO dataset annotations
coco_class_names = {
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

# Open the webcam
cap = cv2.VideoCapture(0)

# Set the frame rate of the video capture
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to torch tensor and normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    input_tensor = transform(frame).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(input_tensor)

    # Draw bounding boxes and labels with different colors for each instance
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score > 0.70:
            box = box.detach().cpu().numpy().astype(int)
            class_name = coco_class_names[int(label)]
            color = (random.randint(100, 255), random.randint(
                100, 255), random.randint(100, 255))

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"{class_name}: {score.item():.2f}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Introduce a slight delay before reading the next frame
    cv2.waitKey(1)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

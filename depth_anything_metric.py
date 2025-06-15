from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained(
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-small-hf"
)
model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-small-hf"
).to(device)


frame = cv2.imread("path_to_your_image.jpg")  # Replace with your image path

image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
inputs = image_processor(images=image, return_tensors="pt", use_fast=True).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)
prediction = prediction.squeeze().cpu().numpy()
prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
prediction = (prediction * 255.0).astype(np.uint8)

cv2.imshow("Depth Prediction", prediction)
key = cv2.waitKey(1)
if key == 27:
    break

cv2.destroyAllWindows()

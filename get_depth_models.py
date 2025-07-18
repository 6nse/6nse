from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import cv2


def get_depth_anything_v2_model(device, size="small", type="Indoor"):
    image_processor = AutoImageProcessor.from_pretrained(
        f"depth-anything/Depth-Anything-V2-Metric-{type}-{size}-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        f"depth-anything/Depth-Anything-V2-Metric-{type}-{size}-hf"
    ).to(device)

    return model, image_processor


def depth_anything_v2_inference(device, model, image_processor, frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = image_processor(images=image, return_tensors="pt", use_fast=True).to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    predicted_depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    predicted_depth = predicted_depth.squeeze().cpu().numpy()
    return predicted_depth


def draw_depth_annotated_image(frame, predicted_depth):
    annotated_image = (predicted_depth - predicted_depth.min()) / (
        predicted_depth.max() - predicted_depth.min()
    )
    annotated_image = (annotated_image * 255.0).astype(np.uint8)

    return annotated_image

from get_depth_models import (
    get_depth_anything_v2_model,
    depth_anything_v2_inference,
    draw_depth_annotated_image,
)
from get_obj_det_models import (
    get_yolo_world_model,
    yolo_world_inference,
    get_florence2_model,
    inference_florence,
    draw_detections_sv,
)

from fastapi import FastAPI, File, UploadFile
from typing import Annotated
import numpy as np
import torch
import cv2
import os

import requests

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_model, depth_image_processor = get_depth_anything_v2_model(device)
object_detection_model, obj_detection_processor = get_florence2_model(device)

app = FastAPI()


@app.post("/depth")
async def get_depth(file: Annotated[bytes, File()], centers: list):
    # Decode the image
    nparr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("Image shape:", img.shape)
    depth_values = []
    depth = depth_anything_v2_inference(device, depth_model, depth_image_processor, img)
    for center in centers:
        center = tuple(center)
        print("Center:", center)
        print("Image shape:", img.shape)
        print("Predicted Depth Shape:", depth.shape)
        print("Predicted Depth Value:", depth[center])
        depth_values.append(depth[center])
    return {"predicted_depth": depth_values}


@app.post("/phase_grounding")
async def get_phase_grounding(file: Annotated[bytes, File()]):
    # Decode the image
    nparr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("Image shape:", img.shape)

    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_input = "A man."
    detections = inference_florence(
        img,
        object_detection_model,
        obj_detection_processor,
        task_prompt,
        text_input,
        device,
    )

    detections_list = []
    for detection in detections:
        x1, y1, x2, y2 = detection[0]
        detections_list.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": detection[5],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            }
        )

    return {"detections": detections_list}


@app.post("/phase_grounding_and_depth_estimation")
async def get_phase_grounding_and_depth_estimation(file: Annotated[bytes, File()]):
    # Decode the image
    depth = requests.post("http://localhost:8000/depth", files={"file": file})
    grounding_detection = requests.post(
        "http://localhost:8000/phase_grounding", files={"file": file}
    )

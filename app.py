from get_depth_models import (
    get_depth_anything_v2_model,
    depth_anything_v2_inference,
)
from get_obj_det_models import (
    get_florence2_model,
    inference_florence,
)

from fastapi import FastAPI, File, UploadFile, Form
from typing import Annotated
import numpy as np
import torch
import cv2
import os
import json
import requests

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_model, depth_image_processor = get_depth_anything_v2_model(
    device, size="large", type="Indoor"
)
object_detection_model, obj_detection_processor = get_florence2_model(device)

app = FastAPI()


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

    return {"grounding_detection": detections_list}


@app.post("/phase_grounding_and_depth_estimation")
async def get_phase_grounding_and_depth_estimation(
    file: Annotated[bytes, File()], text_input: Annotated[str, Form()]
):
    nparr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_input = text_input
    detections = inference_florence(
        img,
        object_detection_model,
        obj_detection_processor,
        task_prompt,
        text_input,
        device,
    )

    print("Detections:", detections)

    detections_list = []
    for detection in detections:
        x1, y1, x2, y2 = detection[0]
        detections_list.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": detection[5],
                "center": (int((y1 + y2) / 2), int((x1 + x2) / 2)),
            }
        )

    print("Detections List:", detections_list)
    depth = depth_anything_v2_inference(device, depth_model, depth_image_processor, img)
    depth_values = []

    for detection in detections_list:
        print("Processing detection:", detection)
        center = detection["center"]
        print("Center:", center)
        depth_values.append(depth[center])
    print("Depth Values:", depth_values)

    for i, detection in enumerate(detections_list):
        detection["depth"] = float(depth_values[i])

    return {"grounding_detection_and_depth": detections_list}

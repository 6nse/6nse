from get_depth_models import get_depth_anything_v2_model, depth_anything_v2_inference
from get_obj_det_models import (
    get_florence2_model,
    inference_florence_od,
    draw_detections_sv,
)

import torch
import cv2
import time

import warnings
import os
from glob import glob

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_model, depth_image_processor = get_depth_anything_v2_model(
    device, size="large", type="Indoor"
)
object_detection_model, obj_detection_processor = get_florence2_model(device)

image_path = "images"  # Replace with your image path

image_list = glob(os.path.join(image_path, "*.jpg"))
start_time = time.time()
for i, image_path in enumerate(image_list):
    print("Loading image...")

    frame = cv2.imread(image_path)
    after_reading = time.time()
    print("Time taken to load image:", after_reading - start_time)

    print("predicting depth...")
    predicted_depth = depth_anything_v2_inference(
        device, depth_model, depth_image_processor, frame
    )
    after_depth = time.time()
    print("time taken to predict depth:", after_depth - after_reading)

    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_input = "A man."
    detections = inference_florence_od(
        frame,
        object_detection_model,
        obj_detection_processor,
        task_prompt,
        text_input,
        device,
    )
    annotated_frame = draw_detections_sv(frame, detections)

    for detection in detections:
        x1, y1, x2, y2 = detection[0]
        center = (int((y1 + y2) / 2), int((x1 + x2) / 2))
        print("Center:", center)
        print("frame shape:", frame.shape)
        print("Predicted Depth Shape:", predicted_depth.shape)
        print("Predicted Depth Value:", predicted_depth[center])
        depth_value = predicted_depth[center]
        cv2.putText(
            annotated_frame,
            f"Depth: {depth_value * 100:.2f}",
            center[::-1],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            3,
        )

        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken: {time_taken:.2f}")
        cv2.imwrite(f"images/annotated_frame_{i}.jpg", annotated_frame)

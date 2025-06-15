from get_depth_models import get_depth_anything_v2_model, depth_anything_v2_inference
from get_obj_det_models import (
    get_yolo_world_model,
    yolo_world_inference,
    get_florence2_model,
    inference_florence_od,
    draw_detections_sv,
)

import torch
import cv2
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

depth_model, depth_image_processor = get_depth_anything_v2_model(
    device, size="small", type="Indoor"
)
object_detection_model, obj_detection_processor = get_florence2_model(device)

cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    predicted_depth = depth_anything_v2_inference(
        device, depth_model, depth_image_processor, frame
    )

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
    # detections = yolo_world_inference(object_detection_model, frame)
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
            (0, 0, 0),
            3,
        )

    cv2.imshow("Object Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    end_time = time.time()
    fps = 1.0 / (end_time - start_time + 1e-8)  # prevent div by zero
    print(f"FPS: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()

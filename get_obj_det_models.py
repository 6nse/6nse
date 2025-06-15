from ultralytics import YOLOWorld
import supervision as sv
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import cv2
import logging

logging.basicConfig(level=logging.DEBUG)


def get_florence2_model(device):
    model = (
        AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        )
        .to(device)
        .eval()
    )

    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True
    )

    return model, processor


def get_yolo_world_model():
    model = YOLOWorld("yolov8x-worldv2.pt")
    return model, None


def yolo_world_inference(model, frame):
    results = model.predict(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    return detections


def draw_detections_sv(image, detections):
    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = bbox_annotator.annotate(scene=image, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections
    )

    return annotated_frame


def inference_florence(frame, model, processor, task_prompt, text_prompt, device):
    if text_prompt is not None:
        prompt = task_prompt + text_prompt
    else:
        prompt = task_prompt
    print("Task Prompt:", task_prompt)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=prompt, images=pil_img, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(generated_text)
    parsed = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=pil_img.size
    )

    # Convert to BoundingBox detections
    detections = sv.Detections.from_lmm(
        sv.LMM.FLORENCE_2, parsed, resolution_wh=pil_img.size
    )
    print(detections)
    return detections

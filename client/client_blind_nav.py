import traceback
import cv2
import requests
import time
import numpy as np

url_depth = "https://cl39uuhdwxc25d-8000.proxy.runpod.net/depth"
url_phase_grounding_and_depth_estimation = (
    "https://cl39uuhdwxc25d-8000.proxy.runpod.net/phase_grounding_and_depth_estimation"
)
headers = {"accept": "application/json"}

cap = cv2.VideoCapture(0)  # 0 for default camera
text_input = "pot with plants?"


while True:
    ret, frame = cap.read()
    if not ret:
        break
    _, img_encoded = cv2.imencode(".jpg", frame)
    img_bytes = img_encoded.tobytes()

    files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}

    try:
        response_depth = requests.post(url_depth, headers=headers, files=files)
        response_phase = requests.post(
            url_phase_grounding_and_depth_estimation,
            headers=headers,
            files=files,
            data={"text_input": text_input},
        )
        print(response_phase.status_code)
        if response_phase.status_code == 200:
            gdd = response_phase.json()["grounding_detection_and_depth"]

        if response_depth.status_code == 200:
            nparr = np.frombuffer(response_depth.content, np.uint8)
            depth_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            for detection in gdd:
                print(detection)
                bbox = detection["bbox"]
                center = detection["center"]
                depth_value = detection["depth"]

                # Draw bounding box
                cv2.rectangle(
                    depth_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                )
                cv2.putText(
                    depth_img,
                    f"({depth_value:.2f}m)",
                    (center[1], center[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            cv2.imshow("Depth Image", depth_img)
        else:
            print(f"Error: {response_depth.status_code} - {response_depth.text}")
    except Exception as e:
        # Print the error message and traceback full details
        print(f"Error: {e}")

    # Display the frame locally (optional)
    cv2.imshow("Live Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(1)  # Wait 1 second between frames (adjust as needed)

cap.release()
cv2.destroyAllWindows()

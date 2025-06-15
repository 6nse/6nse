import cv2
import requests
import time

url = (
    "https://cl39uuhdwxc25d-8000.proxy.runpod.net/phase_grounding_and_depth_estimation"
)
text_input = "A man and bunch of chairs"

cap = cv2.VideoCapture(0)  # 0 for default camera

headers = {"accept": "application/json"}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Encode frame as JPEG
    _, img_encoded = cv2.imencode(".jpg", frame)
    img_bytes = img_encoded.tobytes()

    files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
    data = {"text_input": text_input}

    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        print(response.status_code)
        print(response.json())  # or handle response content
    except Exception as e:
        print(f"Error: {e}")

    # Display the frame locally (optional)
    cv2.imshow("Live Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(1)  # Wait 1 second between frames (adjust as needed)

cap.release()
cv2.destroyAllWindows()

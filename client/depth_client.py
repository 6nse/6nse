import cv2
import requests
import time
import numpy as np

url = "https://cl39uuhdwxc25d-8000.proxy.runpod.net/depth"
headers = {"accept": "application/json"}

cap = cv2.VideoCapture(0)  # 0 for default camera


while True:
    ret, frame = cap.read()
    if not ret:
        break
    _, img_encoded = cv2.imencode(".jpg", frame)
    img_bytes = img_encoded.tobytes()

    files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}

    try:
        response = requests.post(url, headers=headers, files=files)
        if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            depth_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("Depth Image", depth_img)
        else:
            print(f"Error: {response.status_code} - {response.text}")
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

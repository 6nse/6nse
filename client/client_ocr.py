import cv2
import requests
import time

url = "https://cl39uuhdwxc25d-8000.proxy.runpod.net/ocr"
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
        print(response.json())  # or handle response content
        cv2.putText(
            frame,
            "OCR Result: " + str(response.json()),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
    except Exception as e:
        print(f"Error: {e}")

    cv2.imshow("Live Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(1)  # Wait 1 second between frames (adjust as needed)

cap.release()
cv2.destroyAllWindows()

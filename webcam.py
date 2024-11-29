from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Replace with your YOLO model path

# Video capture (use 0 for the webcam)

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # Read a frame from the webcam
        if not success:
            break

        # Run YOLO detection
        results = model.predict(source=frame, save=False, conf=0.5, show=False)

        # Draw bounding boxes for "person" class only
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])  # Class ID
                label = model.names[class_id]  # Class label

                if label == "person":  # Check if the class is "person"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0]  # Confidence score

                    # Draw the bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame in the format required by Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Load HTML for browser

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

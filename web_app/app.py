from ultralytics import YOLO
from flask import Flask, render_template, request
import os
import uuid
import shutil
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model
model = YOLO("models/my_yolov8_model.pt")

# Predefined color map
COLOR_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
    "white": (255, 255, 255)
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            color_choice = request.form.get("box_color", "red")
            color = COLOR_MAP[color_choice]
            # Save uploaded file
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Run prediction (no saving to disk)
            results = model.predict(source=filepath, save=False)
            detections = results[0]
            num_detections = len(detections.boxes)
            if num_detections == 0:
                detection_message = "No objects detected."
            else:
                detection_message = f"{num_detections} object(s) detected."

            # Access the image with bounding boxes (rendered)
            #rendered_img = results[0].plot()  # returns a numpy array (BGR)

            # Load original image
            image = cv2.imread(filepath)

            for box in detections.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{detections.names[cls_id]} {conf:.2f}"

                # Draw box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), color,  thickness=4)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

            # Save manually to static/uploads
            result_filename = f"result_{uuid.uuid4().hex}.jpg"
            result_path = os.path.join(UPLOAD_FOLDER, result_filename)
            cv2.imwrite(result_path, image  )

            return render_template("index.html", result_image=result_filename, detection_message=detection_message)
    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

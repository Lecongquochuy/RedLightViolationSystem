from flask import Flask, Response, render_template, request
import cv2
import os
from plate import LicensePlateProcessor
from objectTracking import objectTracking

app = Flask(__name__)

# Khởi tạo đối tượng xử lý biển số
license_plate_processor = LicensePlateProcessor()

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        file = request.files['video']
        if file:
            # Lưu video tạm thời
            video_path = "temp_video.mp4"
            file.save(video_path)
            return render_template("index.html", video_uploaded=True)
    return render_template("index.html", video_uploaded=False)

def generate_frames(video_path):
    object_tracking = objectTracking(model_path=r"weights\best_addLight.pt", plate_model_path=r"weight_plate/yolo_plate.pt")
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Xử lý frame
        frame, results_tracker, check_red_light = object_tracking.process_frame(frame, count=0)
        frame = object_tracking.track_object(frame, results_tracker, check_red_light)

        # Phát hiện biển số và cắt ảnh
        plates = object_tracking.plate_model.predict(frame, conf=0.3, iou=0.5)
        for box in plates[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_plate = license_plate_processor.crop_license_plate(frame, (x1, y1, x2, y2))

            # Lưu ảnh biển số đã cắt
            plate_image_path = os.path.join("static", "license_plate.jpg")
            cv2.imwrite(plate_image_path, cropped_plate)

            # Nhận diện ký tự trên biển số
            plate_text = license_plate_processor.extract_text_from_plate(cropped_plate)

            # Cập nhật text trực tiếp
            print(f"Detected Plate Text: {plate_text}")  # Debug trên terminal
            with open(os.path.join("static", "plate_text.txt"), "w") as text_file:
                text_file.write(plate_text)

        # Chuyển frame sang JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Trả frame dưới dạng chuỗi byte
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/video-feed")
def video_feed():
    video_path = "temp_video.mp4"
    return Response(
        generate_frames(video_path),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={"Cache-Control": "no-cache"}
    )

@app.route("/get-plate-text")
def get_plate_text():
    try:
        with open(os.path.join("static", "plate_text.txt"), "r") as text_file:
            plate_text = text_file.read()
    except FileNotFoundError:
        plate_text = "No Plate Detected"
    return plate_text

if __name__ == '__main__':
    app.run(debug=True)

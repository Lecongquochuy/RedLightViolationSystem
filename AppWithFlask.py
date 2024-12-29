from flask import Flask, Response, render_template, request
import cv2

from yolov10.objectTracking import objectTracking

app = Flask(__name__)

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
    object_tracking = objectTracking(model_path=r"weights\best_addLight.pt")
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, results_tracker, check_red_light = object_tracking.process_frame(frame, count=0)
        frame = object_tracking.track_object(frame, results_tracker, check_red_light)

        # Chuyển frame sang JPEG
        # cv2.imshow("Frame", frame)
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

if __name__ == '__main__':
    app.run(debug=True)
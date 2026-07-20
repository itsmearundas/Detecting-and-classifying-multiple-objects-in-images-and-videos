from flask import Flask, render_template, request, jsonify
import os
import cv2
from dotenv import load_dotenv
from utils import process_image, process_video, DEFAULT_THRESHOLD, ALL_SUPERCLASSES
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)

# Restrict CORS via CORS_ORIGINS="https://example.com,https://foo.com" in production.
# Defaults to "*" so the app works out of the box on any hosting platform.
_origins = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=_origins.split(",") if _origins != "*" else "*")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configurable upload cap (bytes). Defaults to 500 MB; lower this on
# memory-constrained free-tier hosts via MAX_CONTENT_LENGTH_MB.
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH_MB", 500)) * 1024 * 1024


@app.route("/health")
def health():
    """Lightweight health check for uptime monitors / hosting platforms."""
    return jsonify(status="ok"), 200


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/image", methods=["GET", "POST"])
def image_detection():
    detections            = []
    image_path            = None
    image_path_static     = None
    result                = None
    error_msg             = None
    selected_superclasses = ALL_SUPERCLASSES[:]   # default: all selected

    if request.method == "POST":
        try:
            # ── Parse superclass selection ─────────────────────────────────
            chosen = request.form.getlist("superclass")
            if chosen:
                selected_superclasses = chosen

            # ── Resolve image ──────────────────────────────────────────────
            if "image" in request.files and request.files["image"].filename != "":
                file      = request.files["image"]
                safe_name = file.filename.replace(" ", "_")
                image_path        = os.path.join(UPLOAD_FOLDER, safe_name)
                image_path_static = f"uploads/{safe_name}"
                file.save(image_path)
            else:
                image_path_static = request.form.get("image_path", "").strip()
                if image_path_static:
                    image_path = os.path.join("static", image_path_static)

            # ── Run detection ──────────────────────────────────────────────
            if image_path and os.path.exists(image_path):
                output_img, detections = process_image(
                    image_path,
                    threshold             = DEFAULT_THRESHOLD,
                    selected_superclasses = selected_superclasses,
                )
                cv2.imwrite(os.path.join(UPLOAD_FOLDER, "result.jpg"), output_img)
                result = "uploads/result.jpg"
            else:
                error_msg = "No image found. Please upload a photo first."

        except Exception as exc:
            error_msg = f"An error occurred: {exc}"

    return render_template(
        "image.html",
        result                = result,
        detections            = detections,
        image_path            = image_path_static,
        error_msg             = error_msg,
        all_superclasses      = ALL_SUPERCLASSES,
        selected_superclasses = selected_superclasses,
    )


@app.route("/video", methods=["GET", "POST"])
def video_detection():
    result                = None
    summary               = []
    video_info            = {}
    video_path_static     = None
    error_msg             = None
    selected_superclasses = ALL_SUPERCLASSES[:]

    if request.method == "POST":
        try:
            # ── Parse superclass selection ─────────────────────────────────
            chosen = request.form.getlist("superclass")
            if chosen:
                selected_superclasses = chosen

            if "video" in request.files and request.files["video"].filename != "":
                file      = request.files["video"]
                safe_name = file.filename.replace(" ", "_")
                video_path        = os.path.join(UPLOAD_FOLDER, safe_name)
                video_path_static = f"uploads/{safe_name}"
                file.save(video_path)

                _, summary, video_info = process_video(
                    video_path,
                    threshold             = DEFAULT_THRESHOLD,
                    selected_superclasses = selected_superclasses,
                )
                result = "uploads/result_video.mp4"
            else:
                error_msg = "No video found. Please upload a video file first."

        except Exception as exc:
            error_msg = f"An error occurred: {exc}"

    return render_template(
        "video.html",
        result                = result,
        summary               = summary,
        video_info            = video_info,
        video_path            = video_path_static,
        error_msg             = error_msg,
        all_superclasses      = ALL_SUPERCLASSES,
        selected_superclasses = selected_superclasses,
    )


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
from flask import Flask, request, jsonify
import os
import time

app = Flask(__name__)

UPLOAD_DIR = "backend_snaps"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/plate_event", methods=["POST"])
def plate_event():
    try:
        plate_text = request.form.get("plate_text", "")
        plate_type = request.form.get("plate_type", "")
        timestamp = request.form.get("timestamp", str(int(time.time())))

        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image = request.files["image"]
        filename = f"{plate_text.replace(' ', '_')}_{timestamp}.jpg"
        save_path = os.path.join(UPLOAD_DIR, filename)
        image.save(save_path)

        print("📥 Plate Event Received")
        print("Plate:", plate_text)
        print("Type:", plate_type)
        print("Saved:", save_path)

        return jsonify({
            "status": "success",
            "plate_text": plate_text,
            "plate_type": plate_type,
            "image_path": save_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

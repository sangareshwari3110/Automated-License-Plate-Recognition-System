# ================================================================
# LICENSE PLATE DETECTION
# ================================================================


import cv2
import argparse
import smtplib
import ssl
from email.message import EmailMessage
from ultralytics import YOLO
import math
import time
from google.cloud import vision
import re
import requests
import os
from datetime import datetime


BACKEND_API_URL = "http://127.0.0.1:5000/plate_event"


SNAP_DIR = "lisnaps"
def sanitize_header(text):
    """
    Remove newline and carriage return characters from header strings
    """
    return text.replace("\n", " ").replace("\r", " ").strip()


def save_snap(plate_crop, plate_text, snap_enabled=True):
    """
    Save a clear snapshot of the license plate if snap_enabled is True
    """
    if not snap_enabled or not plate_text:
        return None
    
    
    # Get today's date in dd-mm-yy format
    today = datetime.now().strftime("%d-%m-%y")

    if not os.path.exists(SNAP_DIR):
        os.makedirs(SNAP_DIR)
        
     # Create folder path for today's date
    folder = os.path.join(SNAP_DIR, today)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Sanitize plate text for filename
    safe_text = plate_text.replace(" ", "_").replace("(", "").replace(")", "")
    timestamp = int(time.time())
    filename = os.path.join(folder, f"{safe_text}_{timestamp}.jpg")

    # Save image only if it's not too blurry
    if is_clear_image(plate_crop):
        cv2.imwrite(filename, plate_crop)
        print(f"📸 Snap saved: {filename}")
        return filename
    else:
        print(f"⚠️ Plate image too blurry, snap not saved: {plate_text}")


def is_clear_image(image, threshold=100.0):
    """
    Check if image is clear (not blurry) using Laplacian variance
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()
    return variance > threshold


# Dictionary for Indian state codes
indian_states = {
    'AN': 'Andaman and Nicobar', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
    'AS': 'Assam', 'BR': 'Bihar', 'CH': 'Chandigarh', 'CG': 'Chhattisgarh',
    'DN': 'Dadra and Nagar Haveli', 'DD': 'Daman and Diu', 'DL': 'Delhi',
    'GA': 'Goa', 'GJ': 'Gujarat', 'HR': 'Haryana', 'HP': 'Himachal Pradesh',
    'JK': 'Jammu and Kashmir', 'JH': 'Jharkhand', 'KA': 'Karnataka', 'KL': 'Kerala',
    'LA': 'Ladakh', 'LD': 'Lakshadweep', 'MP': 'Madhya Pradesh', 'MH': 'Maharashtra',
    'MN': 'Manipur', 'ML': 'Meghalaya', 'MZ': 'Mizoram', 'NL': 'Nagaland',
    'OD': 'Odisha', 'PY': 'Puducherry', 'PB': 'Punjab', 'RJ': 'Rajasthan',
    'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TS': 'Telangana', 'TR': 'Tripura',
    'UP': 'Uttar Pradesh', 'UK': 'Uttarakhand', 'WB': 'West Bengal'
}

# ------------------------------
# Command-line Arguments
# ------------------------------
parser = argparse.ArgumentParser(description="License Plate Detection System")

# default mode = "motion"
# default zone = "line"
parser.add_argument(
    "--mode",
    type=str,
    default="motion",
    choices=["static", "motion"],
    help="Detection mode: static or motion"
)

parser.add_argument(
    "--plate",
    type=str,
    default="all",
    choices=["car", "bike", "truck", "all", "license_plate"],
    help="Plate type to detect"
)

parser.add_argument(
    "--zone",
    type=str,
    default="line",
    choices=["off", "rect", "line"],
    help="Zone type: rect for ROI rectangle, line for line crossing detection"
)

parser.add_argument(
    "--mark",
    type=str,
    default="enable",
    choices=["enable", "disable"],
    help="Dynamic marking: enable (draw box) or disable (no drawing)"
)

parser.add_argument(
    "--alert",
    type=str,
    default="none",
    choices=["email", "none"],
    help="Alert mode: email notification or none"
)

# ❌ NO snap argument anymore – always on inside code


def send_to_backend(image_path, plate_text, plate_type):
    try:
        print("send backend")
        with open(image_path, "rb") as img:
            files = {
                "image": img
            }
            data = {
                "plate_text": plate_text,
                "plate_type": plate_type,
                "timestamp": int(time.time())
            }
            r = requests.post(BACKEND_API_URL, files=files, data=data, timeout=5)
            if r.status_code == 200:
                print("✅ Sent to backend")
            else:
                print("❌ Backend error:", r.text)
    except Exception as e:
        print("❌ Backend send failed:", e)


args = parser.parse_args()

# ------------------------------
# Load YOLO Model
# ------------------------------
model = YOLO("license_plate_detector.onnx", verbose=False)   # Replace with your trained plate model
class_names = ["license_plate"]  # Adjust to your dataset classes

# ------------------------------
client = vision.ImageAnnotatorClient.from_service_account_file("apikey.json")
seen_plates = set()  # Track detected plates (to avoid repeated emails)

# Function to extract Indian state from license plate
def get_indian_state(text):
    match = re.match(r"([A-Z]{2})\d{1,2}[A-Z]{0,2}\d{4}", text.replace(" ", "").upper())
    if match:
        code = match.group(1)
        return indian_states.get(code, "Unknown")
    return "Unknown"

def perform_ocr(image_array):
    _, buffer = cv2.imencode(".jpg", image_array)
    content = buffer.tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description.strip().upper()
    return ""

# ------------------------------
# Motion Detection Variables
# ------------------------------
prev_frame = None

def detect_motion(frame, threshold=5000):
    global prev_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_frame is None:
        prev_frame = gray
        return False
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    motion_pixels = cv2.countNonZero(thresh)
    prev_frame = gray
    return motion_pixels > threshold

# ------------------------------
# ROI / Line Drawing
# ------------------------------
roi_rect = None
line_points = []
drawing = False
ix, iy = -1, -1

def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, roi_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # normalize coordinates
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        roi_rect = (x1, y1, x2, y2)
        cv2.rectangle(param, (x1, y1), (x2, y2), (255, 0, 0), 2)

def draw_line(event, x, y, flags, param):
    global drawing, line_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        line_points = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        line_points.append((x, y))
        cv2.line(param, line_points[0], line_points[1], (0, 0, 255), 2)

# ------------------------------
# Email Notification
# ------------------------------
EMAIL_SENDER = "sangareshwari3110@gmail.com"
EMAIL_PASSWORD = "cjersdzjyuytwztg"
EMAIL_RECEIVER = "sangareshwari3110@gmail.com"

def send_email_alert(frame, plate_text):
    try:
        filename = "detected_plate.jpg"
        cv2.imwrite(filename, frame)
        msg = EmailMessage()
        msg["Subject"] = f"License Plate Detected"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content(f"Plate detected: {plate_text}")
        with open(filename, "rb") as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=filename)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("✅ Email alert sent!")
    except Exception as e:
        print("❌ Email failed:", e)

# ------------------------------
# Geometry helpers
# ------------------------------
def point_distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def rect_center(x1, y1, x2, y2):
    return ((x1+x2)//2, (y1+y2)//2)

def on_segment(p, q, r):
    # Check if q lies on pr (collinear and within bounding box)
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
        min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    # Returns orientation of ordered triplet (p, q, r)
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def segments_intersect(p1, q1, p2, q2):
    # General segment intersection test
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False

def rect_intersects_segment(rx1, ry1, rx2, ry2, sx1, sy1, sx2, sy2):
    # Normalize rect coordinates
    left = min(rx1, rx2)
    right = max(rx1, rx2)
    top = min(ry1, ry2)
    bottom = max(ry1, ry2)
    # If either segment endpoint inside rect -> intersect
    if left <= sx1 <= right and top <= sy1 <= bottom:
        return True
    if left <= sx2 <= right and top <= sy2 <= bottom:
        return True
    # Check intersection with any of the 4 rect sides
    rect_edges = [
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top))
    ]
    seg_p = (sx1, sy1)
    seg_q = (sx2, sy2)
    for (p, q) in rect_edges:
        if segments_intersect(p, q, seg_p, seg_q):
            return True
    return False

# ------------------------------
# Tracking subsystem
# ------------------------------
tracked_objects = {}
next_track_id = 0
MAX_MISSING_FRAMES = 15  # remove track after this many frames missing

def iou(boxA, boxB):
    # simple IoU for matching
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def match_track(box, center):
    # match incoming detection to existing tracked_objects using IoU or center distance
    best_id = None
    best_score = 0.0
    for tid, info in tracked_objects.items():
        iou_score = iou(box, info['bbox'])
        center_dist = point_distance(center, info['center'])
        # prefer high IoU, but allow center distance fallback
        score = iou_score + max(0, 1 - (center_dist / 200.0)) * 0.3
        if score > best_score and (iou_score > 0.1 or center_dist < 80):
            best_score = score
            best_id = tid
    return best_id

frame_idx = 0

def start_new_track(box, center, plate_type, frame, x1, y1, x2, y2):
    global next_track_id
    track_id = next_track_id
    next_track_id += 1
    tracked_objects[track_id] = {
        'bbox': box,
        'center': center,
        'last_seen': frame_idx,
        'frames_missing': 0,
        'plate_text': None,
        'ocr_done': False,
        'created_at': time.time(),
        'plate_type': plate_type,
        'snap_sent': False
    }

    # Run OCR immediately on cropped plate
    plate_crop = frame[y1:y2, x1:x2]
    try:
        plate_text = perform_ocr(plate_crop).strip().upper()
    except Exception as e:
        plate_text = ""
        print("OCR error:", e)

    # Get the Indian state from the plate text
    if plate_text:
        state_name = get_indian_state(plate_text)
        plate_text = f"{plate_text} ({state_name})" if state_name != "Unknown" else plate_text
    else:
        plate_text = ""

    # We don't save here to ensure we save after bbox/text are drawn in main loop
    if plate_text:
        tracked_objects[track_id]['plate_text'] = plate_text
        tracked_objects[track_id]['ocr_done'] = True
        if plate_text not in seen_plates:
            seen_plates.add(plate_text)
            if args.alert == "email":
                send_email_alert(frame, plate_text)
    else:
        tracked_objects[track_id]['plate_text'] = None

# ------------------------------
# Main Loop
# ------------------------------
cap = cv2.VideoCapture("licensevideo.mp4")  # Replace with webcam: 0
cv2.namedWindow("LPD")

# --- Draw ROI or Line ---
if args.zone == "rect":
    print("Draw ROI rectangle with mouse drag...")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from video/camera for ROI drawing.")
    scale = 0.5
    frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)
    clone = frame_resized.copy()
    cv2.setMouseCallback("LPD", draw_rect, param=clone)
    while True:
        cv2.imshow("LPD", clone)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

elif args.zone == "line":
    print("Draw line with mouse drag...")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from video/camera for line drawing.")
    scale = 0.5
    frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)
    clone = frame_resized.copy()
    cv2.setMouseCallback("LPD", draw_line, param=clone)
    while True:
        cv2.imshow("LPD", clone)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

print("🚀 Starting License Plate Detection...")

prev_positions = {}  # legacy but not heavily used now

while True:
    ret, frame = cap.read()
    if not ret:
        break

    motion_detected = detect_motion(frame)
    run_detection = True

    if args.mode == "static" and motion_detected:
        run_detection = False
    if args.mode == "motion" and not motion_detected:
        run_detection = False

    scale = 0.5
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    display_frame = frame.copy()
    frame_idx += 1

    # Update missing frame counters for tracked objects (we will set frames_missing=0 when seen)
    for tid in list(tracked_objects.keys()):
        tracked_objects[tid].setdefault('frames_missing', 0)
        tracked_objects[tid]['frames_missing'] += 1
        if tracked_objects[tid]['frames_missing'] > MAX_MISSING_FRAMES:
            # remove stale tracks
            tracked_objects.pop(tid, None)

    if run_detection:
        results = model(frame, conf=0.5,verbose=False)
        detections = []  # collect detections for matching/tracking: [(box, cls_id, plate_type, cx,cy)]
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                plate_type = class_names[cls_id] if cls_id < len(class_names) else "unknown"
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if args.plate != "all" and args.plate != plate_type:
                    continue

                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                detections.append(((x1, y1, x2, y2), cls_id, plate_type, cx, cy))
                
                if args.zone == "rect":
                    # --- Zone Rect Filter ---
                    if args.zone == "rect" and roi_rect:
                        rx1, ry1, rx2, ry2 = roi_rect
                        cx_z, cy_z = (x1 + x2) // 2, (y1 + y2) // 2
                        if not (rx1 < cx_z < rx2 and ry1 < cy_z < ry2):
                            cv2.imshow("LPD", display_frame)
                            continue
                        
                    # --- Extract Plate Region for OCR ---
                    plate_crop = display_frame[y1:y2, x1:x2]
                  
                    try:
                        plate_text = perform_ocr(plate_crop).strip().upper()
                    except Exception as e:
                        plate_text = ""
                        print("OCR error:", e)
                        
                    if plate_text:
                        state_name = get_indian_state(plate_text)
                        if state_name != "Unknown":
                            plate_text = f"{plate_text} ({state_name})"
                    else:
                        plate_text = ""

                    # Draw bbox + text on frame
                    if args.mark == "enable":
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, plate_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    track_info = tracked_objects[track_id]

                    if not track_info['snap_sent']:
                        # Save snapshot ONCE
                        snap_path = save_snap(display_frame, plate_text, snap_enabled=True)
                      
                        # Send to backend
                        send_to_backend(
                            image_path=snap_path,
                            plate_text=plate_text,
                            plate_type=plate_type
                        )

                        track_info['snap_sent'] = True

                    if plate_text and plate_text not in seen_plates:
                        seen_plates.add(plate_text)
                        if args.alert == "email":
                            send_email_alert(display_frame, plate_text)

        # Try to match detections to existing tracks; also check for line intersection to start new track
        for det in detections:
            (x1, y1, x2, y2), cls_id, plate_type, cx, cy = det
            box = (x1, y1, x2, y2)
            center = (cx, cy)

            matched_id = match_track(box, center)
            started_tracking = False

            if matched_id is not None:
                # update existing track
                info = tracked_objects[matched_id]
                info['bbox'] = box
                info['center'] = center
                info['last_seen'] = frame_idx
                info['frames_missing'] = 0
            else:
                # not matched to existing track: check if it intersects the line segment (and only then start tracking)
                if args.zone == "line" and len(line_points) == 2:
                    lx1, ly1 = line_points[0]
                    lx2, ly2 = line_points[1]
                    # Because we drew line on scaled frame, the coordinates match the scaled frame that we are processing.

                    intersects = rect_intersects_segment(x1, y1, x2, y2, lx1, ly1, lx2, ly2)
                    # additional slight touch threshold: check if center is within small distance to the segment
                    if not intersects:
                        # compute distance from center to segment and compare to half height of box (so "touch" counts)
                        px, py = center
                        # segment vector
                        vx = lx2 - lx1
                        vy = ly2 - ly1
                        if vx == 0 and vy == 0:
                            dist_to_seg = point_distance(center, (lx1, ly1))
                        else:
                            # projection t
                            t = ((px - lx1) * vx + (py - ly1) * vy) / (vx*vx + vy*vy)
                            t = max(0, min(1, t))
                            proj_x = lx1 + t * vx
                            proj_y = ly1 + t * vy
                            dist_to_seg = math.hypot(px - proj_x, py - proj_y)
                        half_h = (y2 - y1) / 2.0
                        # if center is close enough to the line (and x projection inside segment), consider as touching
                        if dist_to_seg <= max(8, half_h * 0.6):
                            intersects = True

                    if intersects:
                        # start a new track for this detection
                        track_id = next_track_id
                        next_track_id += 1
                        tracked_objects[track_id] = {
                            'bbox': box,
                            'center': center,
                            'last_seen': frame_idx,
                            'frames_missing': 0,
                            'plate_text': None,
                            'ocr_done': False,
                            'created_at': time.time(),
                            'plate_type': plate_type
                        }
                        started_tracking = True

                        # Run OCR immediately for the triggered object
                        plate_crop = frame[y1:y2, x1:x2]
                        try:
                            plate_text = perform_ocr(plate_crop).strip().upper()
                        except Exception as e:
                            plate_text = ""
                            print("OCR error:", e)
                            
                        if plate_text:
                            state_name = get_indian_state(plate_text)
                            if state_name != "Unknown":
                                plate_text = f"{plate_text} ({state_name})"
                        else:
                            plate_text = ""

                        # Draw bbox + text on frame for this line-triggered detection
                        if args.mark == "enable":
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, plate_text, (x1, max(0, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # ✅ Save FULL FRAME (with bbox + text) when line is crossed
                        snap_path = save_snap(display_frame, plate_text, snap_enabled=True)
                        
                        # Send to backend
                        send_to_backend(
                            image_path=snap_path,
                            plate_text=plate_text,
                            plate_type=plate_type
                        )

                        if plate_text:
                            tracked_objects[track_id]['plate_text'] = plate_text
                            tracked_objects[track_id]['ocr_done'] = True
                            if plate_text not in seen_plates:
                                seen_plates.add(plate_text)
                                if args.alert == "email":
                                    send_email_alert(display_frame, plate_text)
                        else:
                            # no confident OCR result yet; leave plate_text None so we may OCR again later if needed
                            tracked_objects[track_id]['plate_text'] = None

    # Draw all tracked objects on frame (only those started by touching the line)
    for tid, info in tracked_objects.items():
        x1, y1, x2, y2 = info['bbox']
        cx, cy = info['center']
        # Only display if recently seen
        if frame_idx - info.get('last_seen', 0) <= MAX_MISSING_FRAMES:
            if args.mark == "enable":
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                display_text = info.get('plate_text') if info.get('plate_text') else info.get('plate_type', 'plate')
                cv2.putText(display_frame, display_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Optionally: draw the line segment you defined (so user sees the portion)
    if args.zone == "line" and len(line_points) == 2:
        cv2.line(display_frame, line_points[0], line_points[1], (0, 0, 255), 2)
        # draw endpoints
        cv2.circle(display_frame, line_points[0], 4, (0,0,255), -1)
        cv2.circle(display_frame, line_points[1], 4, (0,0,255), -1)

    # --- Show Frame ---
    cv2.imshow("LPD", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

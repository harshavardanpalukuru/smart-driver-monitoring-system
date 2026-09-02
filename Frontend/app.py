from flask import Flask, render_template, redirect, request, jsonify, Response
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()
import mysql.connector
from ultralytics import YOLO
import cv2
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import numpy as np
import base64
import pygame
from collections import defaultdict
from threading import Lock

pygame.mixer.init()

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ================== MySQL Connection ==================
mydb = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME")
)
mycursor = mydb.cursor()

def executionquery(query, values):
    mycursor.execute(query, values)
    mydb.commit()

def retrivequery1(query, values):
    mycursor.execute(query, values)
    return mycursor.fetchall()

def retrivequery2(query):
    mycursor.execute(query)
    return mycursor.fetchall()

# ================== Global Variables ==================
user_email = None
alert_played = False
SOUND_ENABLED = True

# Load YOLOv10 model
model = YOLO('best.pt')

# ================== CORRECT CLASS NAMES ==================
class_names = ['Distracted', 'Drinking', 'Drowsy', 'Eating', 'PhoneUse', 'SafeDriving', 'Seatbelt', 'Smoking']

UNSAFE_BEHAVIORS = ['Distracted', 'Drinking', 'Drowsy', 'Eating', 'PhoneUse', 'Smoking']

# Global detection status (kept for backend but NOT drawn on frames)
current_detection_status = {
    "status": "inactive",
    "safe_driving": True,
    "drowsiness_detected": False,
    "phone_usage": False,
    "distraction": False,
    "drinking_detected": False,
    "eating_detected": False,
    "smoking_detected": False,
    "seatbelt_detected": False,
    "unsafe_behaviors": [],
    "message": "Camera not active"
}

camera = None

# ================== Sound Alert ==================
def play_alert():
    if SOUND_ENABLED:
        try:
            sound = pygame.mixer.Sound(os.path.join("static", "alert.wav"))
            sound.play()
        except Exception as e:
            print(f"Sound error: {e}")

# ================== Consolidated Email Alert ==================
def send_consolidated_alert_email(to_email, unsafe_behaviors_list, source="image"):
    if not unsafe_behaviors_list:
        return False
    
    behaviors_key = ",".join(sorted(unsafe_behaviors_list))
    current_time = time.time()
    
    if current_time - recent_alerts.get(behaviors_key, 0) < ALERT_COOLDOWN:
        print(f"⏸️ Cooldown active for: {unsafe_behaviors_list}")
        return False
    
    recent_alerts[behaviors_key] = current_time
    
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    
    behavior_display_names = {
        'Distracted': '🚗 Distracted Driving',
        'Drinking': '🥤 Drinking',
        'Drowsy': '😴 Drowsiness',
        'Eating': '🍔 Eating',
        'PhoneUse': '📱 Phone Usage',
        'Smoking': '🚬 Smoking',
        'NoSeatbelt': '🪑 No Seatbelt'
    }
    
    recommendations = {
        'Distracted': '• Focus on the road\n• Avoid any distractions\n• Keep both hands on the wheel',
        'Drinking': '• Do not drink while driving\n• Pull over safely if you need to drink',
        'Drowsy': '• Take a break immediately\n• Pull over at a safe location',
        'Eating': '• Avoid eating while driving\n• Pull over to eat',
        'PhoneUse': '• Put away your phone\n• Use hands-free if necessary',
        'Smoking': '• Avoid smoking while driving\n• Pull over if you need to smoke',
        'NoSeatbelt': '• WEAR YOUR SEATBELT IMMEDIATELY!'
    }
    
    behaviors_list = []
    for behavior in unsafe_behaviors_list:
        display_name = behavior_display_names.get(behavior, behavior)
        rec = recommendations.get(behavior, 'Correct the unsafe behavior immediately')
        behaviors_list.append(f"⚠️ **{display_name}**\n   🔧 Action Required:\n{rec}")
    
    behaviors_text = "\n\n".join(behaviors_list)
    source_text = "📸 Image Upload" if source == "image" else "🎥 Live Camera Feed"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = f"⚠️ SAFETY ALERT: {len(unsafe_behaviors_list)} Unsafe Behavior(s) Detected!"

        body = f"""
🚨 **DRIVER SAFETY ALERT** 🚨

📌 Source: {source_text}
⏰ Time: {time.strftime('%d-%m-%Y %H:%M:%S')}
👤 Driver: {to_email}
🔴 Total Unsafe Behaviors: {len(unsafe_behaviors_list)}

DETECTED BEHAVIORS:
{behaviors_text}

⚠️ IMMEDIATE ACTION REQUIRED
Please address ALL unsafe behaviors immediately.

Stay Alert & Drive Safely!
— Smart Driver Monitoring System
"""
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        
        print(f"✅ Email sent for {len(unsafe_behaviors_list)} behaviors: {unsafe_behaviors_list}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False

# Global for email cooldown
recent_alerts = defaultdict(float)
ALERT_COOLDOWN = 30

# ================== CLEAN FRAME PROCESSING (ONLY BOUNDING BOXES + LABELS) ==================
def process_frame_clean(frame):
    """
    Draws ONLY bounding boxes with labels.
    NO status text, NO "UNSAFE" warnings, NO seatbelt text, NO extra overlays.
    Pure detection visualization.
    """
    annotated = frame.copy()
    unsafe_detected = []
    seatbelt_detected = False

    # Run inference
    results = model(frame, conf=0.18, imgsz=640, verbose=False)[0]

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = class_names[cls]

            # Track seatbelt and unsafe behaviors for email alerts (not drawn)
            if label == 'Seatbelt':
                seatbelt_detected = True
            if label in UNSAFE_BEHAVIORS:
                unsafe_detected.append(label)

            # Color: Red for unsafe behaviors, Green for safe behaviors
            is_unsafe = label in UNSAFE_BEHAVIORS
            color = (0, 0, 255) if is_unsafe else (0, 255, 0)

            # Draw bounding box (thicker for visibility)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)

            # Prepare label text
            label_text = f"{label} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Position label above the box
            label_y = max(y1 - 10, text_h + 10)
            label_x = x1
            
            # Draw background rectangle for text
            cv2.rectangle(annotated, (label_x, label_y - text_h - 8),
                          (label_x + text_w + 10, label_y + 4), color, -1)
            # Draw text
            cv2.putText(annotated, label_text, (label_x + 5, label_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # No Seatbelt check (for email only, NOT drawn on frame)
    if not seatbelt_detected:
        unsafe_detected.append('NoSeatbelt')
    
    unique_unsafe = list(set(unsafe_detected))
    
    # Return annotated frame (clean) + metadata for alerts
    return annotated, unique_unsafe, seatbelt_detected

# ================== IMAGE ANALYSIS (CLEAN VERSION) ==================
def analyze_image_with_boxes(image_data):
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "error", "message": "Could not decode image"}

        frame = cv2.resize(frame, (640, 480))
        # Use clean processing (only boxes + labels)
        annotated, unique_unsafe, seatbelt_detected = process_frame_clean(frame)

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        result = {
            "status": "success",
            "safe_driving": len([b for b in unique_unsafe if b != 'NoSeatbelt']) == 0,
            "unsafe_behaviors": unique_unsafe,
            "annotated_image": annotated_base64,
            "seatbelt_detected": seatbelt_detected,
            "drowsiness_detected": 'Drowsy' in unique_unsafe,
            "phone_usage": 'PhoneUse' in unique_unsafe,
            "distraction": 'Distracted' in unique_unsafe,
            "drinking_detected": 'Drinking' in unique_unsafe,
            "eating_detected": 'Eating' in unique_unsafe,
            "smoking_detected": 'Smoking' in unique_unsafe,
            "noseatbelt_detected": not seatbelt_detected,
            "total_unsafe_count": len(unique_unsafe)
        }

        # Send email alert if unsafe behaviors detected (background, not shown on image)
        if unique_unsafe and user_email:
            send_consolidated_alert_email(user_email, unique_unsafe, source="image")

        return result

    except Exception as e:
        print(f"Image analysis error: {e}")
        return {"status": "error", "message": str(e)}

# ================== LIVE CAMERA FEED (CLEAN VERSION) ==================
def generate_frames():
    global camera, alert_played, current_detection_status

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(1.0)

    last_unsafe_set = set()
    last_email_time = 0
    frame_counter = 0

    while True:
        success, frame = camera.read()
        if not success:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # In case of camera error, we still show a clean frame with no extra text (just blank)
            cv2.putText(frame, "Camera Error", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            frame = cv2.resize(frame, (640, 480))
        
        # Process frame with clean bounding boxes + labels (NO extra text overlays)
        annotated, unique_unsafe, seatbelt_detected = process_frame_clean(frame)

        # Update global status (for backend, not displayed)
        current_detection_status = {
            "status": "active",
            "safe_driving": len([b for b in unique_unsafe if b != 'NoSeatbelt']) == 0,
            "drowsiness_detected": 'Drowsy' in unique_unsafe,
            "phone_usage": 'PhoneUse' in unique_unsafe,
            "distraction": 'Distracted' in unique_unsafe,
            "drinking_detected": 'Drinking' in unique_unsafe,
            "eating_detected": 'Eating' in unique_unsafe,
            "smoking_detected": 'Smoking' in unique_unsafe,
            "seatbelt_detected": seatbelt_detected,
            "unsafe_behaviors": unique_unsafe,
            "total_unsafe_count": len(unique_unsafe),
            "message": "Detection active (clean view)"
        }

        # Email and Sound Alert (functionality preserved, but not shown on video)
        current_time = time.time()
        if unique_unsafe and user_email:
            curr_set = set(unique_unsafe)
            if curr_set != last_unsafe_set or (current_time - last_email_time) >= ALERT_COOLDOWN:
                send_consolidated_alert_email(user_email, unique_unsafe, source="video")
                last_unsafe_set = curr_set
                last_email_time = current_time

        if unique_unsafe and not alert_played:
            play_alert()
            alert_played = True
        elif not unique_unsafe:
            alert_played = False

        # Encode and stream frame (clean, only boxes+labels)
        ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        frame_counter += 1
        time.sleep(0.033)

# ================== ROUTES ==================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_image', methods=['POST'])
def analyze_image_endpoint():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No image selected"}), 400

    image_bytes = file.read()
    result = analyze_image_with_boxes(image_bytes)
    return jsonify(result)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_status', methods=['GET'])
def detection_status():
    return jsonify(current_detection_status)

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global camera, current_detection_status
    if camera is not None:
        camera.release()
        camera = None
    current_detection_status = {
        "status": "inactive",
        "safe_driving": True,
        "unsafe_behaviors": [],
        "message": "Detection stopped"
    }
    return jsonify({"status": "success"})

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get('name')
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        if password == confirmpassword:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_list = [i[0] for i in email_data]
            if email.upper() not in email_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
                executionquery(query, (name, email, hashed_password))
                return render_template('register.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID already exists!")
        return render_template('register.html', message="Confirm password does not match!")
    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_list = [i[0] for i in email_data]
        if email.upper() in email_list:
            query = "SELECT password FROM users WHERE email = %s"
            password_data = retrivequery1(query, (email,))
            if password_data and check_password_hash(password_data[0][0], password):
                global user_email
                user_email = email
                return redirect("/home")
            return render_template('login.html', message="Invalid Password!!")
        return render_template('login.html', message="This email ID does not exist!")
    return render_template('login.html')

@app.route('/home')
def home():
    if not user_email:
        return redirect('/login')
    return render_template('home.html')

@app.route('/model_info')
def model_info():
    return jsonify({
        "total_classes": len(class_names),
        "classes": class_names,
        "unsafe_behaviors": UNSAFE_BEHAVIORS
    })

if __name__ == '__main__':
    print("=" * 80)
    print("🚗 SMART DRIVER MONITORING SYSTEM - CLEAN VISUAL MODE")
    print("✅ Feature: ONLY bounding boxes + labels displayed")
    print("❌ Removed: Status text, warning overlays, seatbelt text, unsafe count")
    print("📧 Email alerts & sounds still active in background")
    print("Class Names:", class_names)
    print("Confidence Threshold: 0.18")
    print("=" * 80)
    app.run(debug=True)
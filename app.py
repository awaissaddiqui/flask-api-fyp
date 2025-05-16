# --- Final Updated Code ---
import cv2
from ultralytics import YOLO
from flask import Flask, request, render_template, jsonify
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv
import json

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

last_email_sent = {}

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

def connect_to_supabase():
    try:
        SUPABASE_URL = os.getenv("SUPABASE_PROJECT_URL")
        SUPABASE_API_KEY = os.getenv("SUPABASE_KEY")
        if not SUPABASE_URL or not SUPABASE_API_KEY:
            raise ValueError("Supabase URL or API Key missing.")
        return create_client(SUPABASE_URL, SUPABASE_API_KEY)
    except Exception as e:
        print(f"Supabase connection error: {e}")
        return None

supabase = connect_to_supabase()
model = YOLO('best_new.pt')

def send_email(subject, body, recipient):
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient, msg.as_string())
        server.quit()
        print("✅ Email sent to", recipient)
    except Exception as e:
        print("❌ Email send error:", e)

def upload_frame_to_supabase(frame, filename):
    try:
        success, encoded_frame = cv2.imencode('.jpg', frame)
        if not success:
            raise ValueError("Frame encoding failed.")
        frame_bytes = encoded_frame.tobytes()

        supabase.storage.from_("smartcity").upload(
            path=f"public/{filename}",
            file=frame_bytes,
            file_options={"content-type": "image/jpeg", "upsert": False}
        )

        return supabase.storage.from_("smartcity").get_public_url(f"public/{filename}")
    except Exception as e:
        print("❌ Upload error:", e)
        return None

def process_frame(frame, camera_id):
    global last_email_sent
    now = datetime.now()
    detections_confidence = {}
    uploaded_urls = {}
    role_mapping = {
        "fire": "fire_department",
        "smoke": "fire_department",
        "gun": "law_enforcement",
        "accident": "medical_services",
        "pothole": "road_maintenance",
    }

    # Get camera location
    try:
        cam_data = supabase.table("cameras").select("location").eq("id", camera_id).execute()
        if not cam_data.data or len(cam_data.data) == 0:
            print(f"⚠️ Camera ID {camera_id} not found.")
            return
        location = cam_data.data[0]['location']
    except Exception as e:
        print("❌ Error fetching camera info:", e)
        return

    results = model.predict(frame)
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            confidence = float(box.conf) * 100
            if label not in detections_confidence:
                detections_confidence[label] = {"confidence": [], "max": 0, "box": None}
            detections_confidence[label]["confidence"].append(confidence)
            if confidence > detections_confidence[label]["max"]:
                detections_confidence[label]["max"] = confidence
                detections_confidence[label]["box"] = box.xyxy.tolist()[0]
    print(detections_confidence)
    for label, data in detections_confidence.items():
        if data["max"] >= 65:
            filename = f"{label}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            public_url = upload_frame_to_supabase(frame, filename)
            uploaded_urls[label] = public_url

            role = role_mapping.get(label.lower())
            if role:
                try:
                    users = supabase.table("users").select("email").eq("role", role).execute()
                    if not users.data:
                        print(f"⚠️ No users found for role {role}")
                        continue

                    for user in users.data:
                        email = user['email']
                        last_sent = last_email_sent.get((label, email))
                        if not last_sent or (now - last_sent) > timedelta(hours=3):
                            send_email(
                                subject=f"🚨 {label.capitalize()} Alert",
                                body = f"""
                                    🚨 URGENT ALERT

                                    Dear User,

                                    This is an automated notification from the Smart City Surveillance System. An anomaly has been detected in a monitored zone and may require your immediate attention.

                                    📍 Location: {location}  
                                    📌 Detected: {label.capitalize()}  
                                    🎯 Confidence Level: {data['max']:.2f}%

                                    ⚠️ What This Means:
                                    Our system has identified unusual activity {label.capitalize()} with a high confidence level. This may indicate a potential emergency.

                                    🔍 Recommended Actions:
                                    1. Review the live or recorded footage for verification.  
                                    2. If the alert appears to be a false positive, consider adjusting sensitivity settings.  
                                    3. If the threat is confirmed, initiate the appropriate safety or emergency response protocols.

                                    🔒 Please Note:
                                    This is an automated message — replies are not monitored.

                                    Stay Safe,  
                                    Smart City Surveillance System 🤖
                                    """,
                                recipient=email
                            )
                            last_email_sent[(label, email)] = now

                            detection_entry = {
                                "label": label,
                                "confidence": data["max"],
                                "timestamp": now.isoformat(),
                                "location": location,
                                "camera_id": camera_id,
                                "email_sent": True,
                                "authority_email": email,
                                "next_email_allowed_at": (now + timedelta(hours=3)).isoformat(),
                                "frame_url": public_url,
                                "is_highest_confidence": True,
                                "detection_metadata": {
                                    "confidence_array": data["confidence"],
                                    "bounding_box": data["box"]
                                }
                            }
                            res = supabase.table("detections").insert(detection_entry).execute()
                            detection_id = res.data[0]['id'] if res.data else None

                            if detection_id:
                                supabase.table("alerts").insert({
                                    "detection_id": detection_id,
                                    "authority_email": email,
                                    "status": "sent",
                                    "sent_at": now.isoformat(),
                                    "acknowledged_at": None
                                }).execute()
                                print(f"📨 Alert saved for {label} to {email}")
                except Exception as e:
                    print(f"❌ Error processing users for {label}: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test_frame', methods=['POST'])
def test_frame():
    if 'frame' not in request.files:
        return jsonify({"error": "No frame provided"}), 400

    file = request.files['frame']
    camera_id = request.form.get('camera_id')

    if not file or not camera_id:
        return jsonify({"error": "Camera ID and frame required"}), 400

    try:
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 500
        process_frame(frame, camera_id)
        return jsonify({"message": "Frame processed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

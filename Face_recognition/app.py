# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# import os
# import pickle
# from datetime import datetime

# # Load known faces
# def load_known_faces():
#     if os.path.exists("known_faces.pkl"):
#         with open("known_faces.pkl", "rb") as f:
#             return pickle.load(f)
#     return {}

# # Save known faces
# def save_known_faces(known_faces):
#     with open("known_faces.pkl", "wb") as f:
#         pickle.dump(known_faces, f)

# # Mark attendance only once per session
# def mark_attendance(name, marked_names):
#     if name not in marked_names:
#         with open("Attendance.csv", "a") as f:
#             now = datetime.now()
#             dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
#             f.write(f"{name},{dt_string}\n")
#         marked_names.add(name)
#     return marked_names

# # Streamlit UI
# st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
# st.title("📸 Face Recognition Attendance System")

# menu = st.sidebar.selectbox("Menu", ["Register Face", "Live Attendance"])

# known_faces = load_known_faces()

# if menu == "Register Face":
#     st.header("📝 Register a New Face")
#     name = st.text_input("Enter Name:")
#     uploaded_image = st.camera_input("Take a photo to register")

#     if uploaded_image is not None and name:
#         file_bytes = np.asarray(bytearray(uploaded_image.getvalue()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)

#         face_encodings = face_recognition.face_encodings(img)
#         if face_encodings:
#             known_faces[name] = face_encodings[0]
#             save_known_faces(known_faces)
#             st.success(f"{name} registered successfully!")
#         else:
#             st.error("No face detected. Please try again.")

# elif menu == "Live Attendance":
#     st.header("📡 Live Webcam Attendance")
#     run = st.checkbox("Start Camera")

#     FRAME_WINDOW = st.image([])
#     marked_names = set()

#     cap = None
#     if run:
#         cap = cv2.VideoCapture(0)

#     while run:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("❌ Failed to access camera")
#             break

#         # Resize for speed
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#         # Detect faces
#         face_locations = face_recognition.face_locations(rgb_small)
#         face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
#             name = "Unknown"

#             if True in matches:
#                 match_index = matches.index(True)
#                 name = list(known_faces.keys())[match_index]
#                 marked_names = mark_attendance(name, marked_names)

#             # Draw label on frame
#             y1, x2, y2, x1 = [v * 4 for v in face_locations[0]]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, name, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         FRAME_WINDOW.image(frame, channels="BGR")

#     if cap:
#         cap.release()
import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pickle
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ------------------- FACE STORAGE -------------------
def load_known_faces():
    if os.path.exists("known_faces.pkl"):
        with open("known_faces.pkl", "rb") as f:
            return pickle.load(f)
    return {}

def save_known_faces(known_faces):
    with open("known_faces.pkl", "wb") as f:
        pickle.dump(known_faces, f)

# ------------------- ATTENDANCE LOGIC -------------------
PERIODS = [
    ("Period 1", datetime.strptime("09:00", "%H:%M").time(), datetime.strptime("10:00", "%H:%M").time()),
    ("Period 2", datetime.strptime("10:00", "%H:%M").time(), datetime.strptime("11:00", "%H:%M").time()),
    ("Period 3", datetime.strptime("11:00", "%H:%M").time(), datetime.strptime("12:00", "%H:%M").time()),
    ("Period 4", datetime.strptime("12:00", "%H:%M").time(), datetime.strptime("13:00", "%H:%M").time()),
    ("Lunch Break", datetime.strptime("13:00", "%H:%M").time(), datetime.strptime("14:00", "%H:%M").time()),
    ("Period 5", datetime.strptime("14:00", "%H:%M").time(), datetime.strptime("15:00", "%H:%M").time()),
    ("Period 6", datetime.strptime("15:00", "%H:%M").time(), datetime.strptime("16:00", "%H:%M").time()),
]

def init_attendance():
    columns = ["Name"] + [p[0] for p in PERIODS] + ["Last Seen"]
    if not os.path.exists("Attendance.csv"):
        df = pd.DataFrame(columns=columns)
        df.to_csv("Attendance.csv", index=False)
    else:
        df = pd.read_csv("Attendance.csv")
        if list(df.columns) != columns:
            df = pd.DataFrame(columns=columns)
            df.to_csv("Attendance.csv", index=False)

def update_attendance(name, current_time):
    df = pd.read_csv("Attendance.csv")

    if name not in df["Name"].values:
        df.loc[len(df)] = [name] + ["Absent" if p[0] != "Lunch Break" else "N/A" for p in PERIODS] + [str(current_time)]

    idx = df.index[df["Name"] == name][0]
    df.at[idx, "Last Seen"] = str(current_time)

    for i, (period_name, start, end) in enumerate(PERIODS):
        if period_name == "Lunch Break":
            df.at[idx, period_name] = "N/A"
            continue

        if start <= current_time.time() <= end:
            df.at[idx, period_name] = "Present"
        else:
            try:
                last_seen = datetime.strptime(df.at[idx, "Last Seen"], "%Y-%m-%d %H:%M:%S.%f")
            except:
                last_seen = datetime.strptime(df.at[idx, "Last Seen"], "%Y-%m-%d %H:%M:%S")
            if current_time.time() > end and (current_time - last_seen).seconds > 600:
                df.at[idx, period_name] = "Absent"

    df.to_csv("Attendance.csv", index=False)

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("📸 Face Recognition Attendance System")

menu = st.sidebar.selectbox("Menu", ["Register Face", "Live Attendance", "View Attendance", "Analytics Dashboard"])
known_faces = load_known_faces()
init_attendance()

# ------------------- REGISTER FACE -------------------
if menu == "Register Face":
    st.header("📝 Register a New Face")
    name = st.text_input("Enter Name:")
    uploaded_image = st.camera_input("Take a photo to register")

    if uploaded_image is not None and name:
        file_bytes = np.asarray(bytearray(uploaded_image.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        face_encodings = face_recognition.face_encodings(img)
        if face_encodings:
            known_faces[name] = face_encodings[0]
            save_known_faces(known_faces)
            st.success(f"{name} registered successfully!")
        else:
            st.error("No face detected. Please try again.")

# ------------------- LIVE ATTENDANCE -------------------
elif menu == "Live Attendance":
    st.header("📡 Live Webcam Attendance")

    class FaceRecognitionProcessor(VideoProcessorBase):
        def __init__(self):
            self.known_faces = known_faces

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(list(self.known_faces.values()), face_encoding)
                name = "Unknown"
                if True in matches:
                    match_index = matches.index(True)
                    name = list(self.known_faces.keys())[match_index]
                    update_attendance(name, datetime.now())

                y1, x2, y2, x1 = [v * 4 for v in face_location]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            return img

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
        key="face-recognition",
        video_processor_factory=FaceRecognitionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ------------------- VIEW ATTENDANCE -------------------
elif menu == "View Attendance":
    st.header("📊 Attendance Records")
    if os.path.exists("Attendance.csv"):
        df = pd.read_csv("Attendance.csv")
        st.dataframe(df)
    else:
        st.info("No attendance records found yet.")

# ------------------- ANALYTICS DASHBOARD -------------------
elif menu == "Analytics Dashboard":
    st.header("📊 Attendance Analytics")
    if os.path.exists("Attendance.csv"):
        df = pd.read_csv("Attendance.csv")

        if df.empty:
            st.warning("No attendance data available yet.")
        else:
            data = df.drop(columns=["Last Seen"], errors="ignore")

            st.subheader("📌 Raw Attendance Data")
            st.dataframe(data)

            st.subheader("👩‍🎓 Student Attendance Percentage")
            student_summary = data.set_index("Name")
            present_counts = (student_summary == "Present").sum(axis=1)
            total_periods = (student_summary != "Lunch Break").sum(axis=1)
            attendance_percent = (present_counts / total_periods) * 100
            st.bar_chart(attendance_percent)

            st.subheader("⏰ Period Attendance Percentage")
            period_summary = {}
            for col in data.columns[1:]:
                if col != "Lunch Break":
                    period_summary[col] = (data[col] == "Present").mean() * 100
            st.bar_chart(period_summary)

            st.subheader("📄 Summary Table")
            summary_df = pd.DataFrame({"Attendance %": attendance_percent.round(2)})
            st.dataframe(summary_df)
    else:
        st.info("No attendance records found yet.")  


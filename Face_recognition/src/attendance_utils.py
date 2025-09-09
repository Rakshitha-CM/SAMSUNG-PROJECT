import os
import pandas as pd
from datetime import datetime

ATTENDANCE_FILE = os.path.join("data", "attendance.csv")

# ---------------- Ensure File Exists with Headers ----------------
def ensure_attendance_file():
    """Create attendance.csv with headers if missing or empty."""
    if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)

# ---------------- Mark Attendance ----------------
def mark_attendance(name):
    ensure_attendance_file()

    df = pd.read_csv(ATTENDANCE_FILE)

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    # Check if already marked today
    already_marked = ((df["Name"] == name) & (df["Date"] == today)).any()

    if not already_marked:
        new_entry = pd.DataFrame([[name, today, current_time]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

# ---------------- Get Attendance Records ----------------
def get_attendance():
    ensure_attendance_file()
    return pd.read_csv(ATTENDANCE_FILE)

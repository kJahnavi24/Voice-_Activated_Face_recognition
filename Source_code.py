import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime, timedelta
import pyttsx3

# Initialize pyttsx3
engine = pyttsx3.init()

# Set up webcam
video_capture = cv2.VideoCapture(0)

# Set up face recognition
folder_path = "C:/Users/HP/Downloads/image_opencv"
known_face_encoding = []
known_faces_names = []

# Load known face encodings and names from images folder
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        encoding = face_encodings[0]
        known_face_encoding.append(encoding)
        known_faces_names.append(os.path.splitext(filename)[0])

# Print the known face encodings and names
for encoding, name in zip(known_face_encoding, known_faces_names):
    print(name, encoding)

face_locations = []
face_encodings = []
face_names = []

# Set up time variables
previous_time = {}

# Open the CSV file in append mode
current_date = datetime.now().strftime("%Y-%m-%d")
csv_file = open(current_date + '.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)

# Set up time interval variables
interval = timedelta(seconds=5)  # Time interval of 5 seconds
start_time = datetime.now()

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect faces and perform face recognition every 5 seconds
    elapsed_time = datetime.now() - start_time
    if elapsed_time >= interval:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
                engine.say("Welcome, " + name)  # Speak the recognized name
                engine.runAndWait()

            face_names.append(name)

        # Update text and CSV every 5 seconds
        current_time = datetime.now().strftime("%H:%M:%S")
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (10, 100)
        font_scale = 1.5
        font_color = (0, 255, 0)
        thickness = 3
        line_type = 2

        for name in face_names:
            if name not in previous_time or (datetime.now() - previous_time[name]) >= interval:
                cv2.putText(frame, name + ' Present',
                            bottom_left_corner_of_text,
                            font,
                            font_scale,
                            font_color,
                            thickness,
                            line_type)

                csv_writer.writerow([name, current_date, current_time])
                previous_time[name] = datetime.now()

        start_time = datetime.now()  # Reset start time after updating text and CSV

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close CSV file
video_capture.release()
cv2.destroyAllWindows()
csv_file.close()

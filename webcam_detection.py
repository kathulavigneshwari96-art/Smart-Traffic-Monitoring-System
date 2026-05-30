from ultralytics import YOLO
import cv2
import time
import easyocr
import re
import csv

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load OCR reader
reader = easyocr.Reader(['en'])

# Allowed classes
allowed_classes = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'bus',
    'truck'
]

# Vehicle classes
vehicle_classes = [
    'car',
    'motorcycle',
    'bus',
    'truck'
]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not opened")
    exit()

# Video writer setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    'output_video.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (frame_width, frame_height)
)

# Log file
log_file = open("vehicle_log.txt", "a")
csv_file = open(
    "traffic_report.csv",
    "a",
    newline=""
)

csv_writer = csv.writer(csv_file)

# FPS calculation
prev_time = time.time()

# Unique vehicle counting
counted_ids = set()
total_vehicle_count = 0
logged_ids = set()

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Run YOLO detection
    results = model.track(
    frame,
    persist=True,
    verbose=False
)

    vehicle_count = 0
    person_count = 0

    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0


    # Process detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]
            track_id = -1

            if  box.id is not None:
              track_id = int(box.id[0])

            if class_name in allowed_classes and confidence > 0.5:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                if track_id != -1:
                   label = f"{class_name} ID:{track_id}"
                else:
                   label = f"{class_name}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

                # Person count
                if class_name == "person":
                    person_count += 1

                # Vehicle count
                if class_name in vehicle_classes:

                    vehicle_count += 1

                    if track_id != -1 and track_id not in counted_ids:

                       counted_ids.add(track_id)

                       total_vehicle_count += 1
                    if track_id not in logged_ids:

                       csv_writer.writerow([
                           time.strftime("%Y-%m-%d"),
                           time.strftime("%H:%M:%S"),
                           class_name,
                           track_id
                     ])

                    logged_ids.add(track_id)

                    log_file.write(
                        f"{time.ctime()} - {class_name}\n"
                    )

                    # OCR for number plate reading
                    try:

                        vehicle_crop = frame[y1:y2, x1:x2]

                        if vehicle_crop.size > 0:

                            ocr_results = reader.readtext(
                                vehicle_crop
                            )

                            for res in ocr_results:

                                detected_text = res[1]

                                plate_text = re.sub(
                                    r'[^A-Z0-9]',
                                    '',
                                    detected_text.upper()
                                )

                                if len(plate_text) >= 6:

                                    cv2.putText(
                                        frame,
                                        f"Plate: {plate_text}",
                                        (x1, y2 + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 255),
                                        2
                                    )

                                    break

                    except Exception:
                        pass

    # Traffic Density Logic
    if vehicle_count <= 3:
        traffic_status = "Low Traffic"
    elif vehicle_count <= 7:
        traffic_status = "Medium Traffic"
    else:
        traffic_status = "High Traffic"

    # Vehicle count display
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # Traffic status display
    cv2.putText(
        frame,
        traffic_status,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        3
    )

    # FPS calculation
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    # Person count display
    cv2.putText(
        frame,
        f"Persons: {person_count}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        3
    )
    cv2.putText(
    frame,
    f"Total Vehicles: {total_vehicle_count}",
    (20, 200),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3
)

    # Save video
    out.write(frame)

    # Show output
    cv2.imshow(
        "Smart Traffic Vehicle Detection",
        frame
    )

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    # Save screenshot
    if key == ord('s'):

        filename = (
            f"screenshot_{int(time.time())}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"Screenshot saved: {filename}"
        )

    # Quit
    if key == ord('q'):
        break

# Release everything
cap.release()
out.release()
log_file.close()
csv_file.close()

cv2.destroyAllWindows()

print("Video saved successfully!")
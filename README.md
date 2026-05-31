AI-Powered Smart Traffic Monitoring and Analytics System
📌 Project Overview

The AI-Powered Smart Traffic Monitoring and Analytics System is a computer vision project developed using YOLOv8, OpenCV, EasyOCR, and Python. The system performs real-time traffic monitoring through a webcam or video feed by detecting and tracking vehicles and pedestrians.

The project provides traffic analytics, vehicle counting, person counting, traffic density estimation, OCR-based number plate recognition, video recording, screenshot capture, and CSV-based traffic data logging.

🚀 Features
Vehicle Detection
Detects Cars
Detects Motorcycles
Detects Buses
Detects Trucks
Person Detection
Detects pedestrians in real time
Vehicle Tracking
Assigns unique tracking IDs to detected vehicles
Tracks vehicles across video frames
Traffic Analytics
Vehicle Counting
Unique Vehicle Counting
Person Counting
Traffic Density Detection
Number Plate Recognition
OCR-based number plate text extraction using EasyOCR
Data Logging
Stores vehicle information in CSV format
Generates traffic analytics data for future analysis
Video Processing
Real-time webcam monitoring
Video recording and saving
Screenshot capture
Performance Monitoring
Real-time FPS (Frames Per Second) display
🛠️ Technologies Used
Python
YOLOv8
OpenCV
EasyOCR
NumPy
Ultralytics
CSV Data Logging
📂 Project Workflow

Camera Input
↓
YOLOv8 Detection
↓
Vehicle & Person Identification
↓
Vehicle Tracking IDs
↓
Vehicle Counting
↓
Traffic Density Analysis
↓
OCR Number Plate Recognition
↓
CSV Traffic Analytics
↓
Video Recording & Screenshots

📊 Output Information

The system displays:

Vehicle Count
Total Vehicle Count
Person Count
Traffic Density Status
Vehicle Tracking IDs
FPS
Number Plate Information
📁 Generated Files
Output Video
output_video.mp4
Vehicle Logs
vehicle_log.txt
Traffic Analytics
traffic_report.csv
Screenshots
screenshot_timestamp.jpg
▶️ Installation
Clone Repository
git clone https://github.com/https://github.com/kathulavigneshwari96-art/Smart-Traffic-Monitoring-System/edit/main/README.md/Smart-Traffic-Monitoring-System.git
cd Smart-Traffic-Monitoring-System
Install Dependencies
pip install ultralytics
pip install opencv-python
pip install easyocr
pip install numpy
pip install lap

Or install using:

pip install -r requirements.txt
▶️ Run Project
python webcam_detection.py
🎮 Controls
Key	Action
S	Save Screenshot
Q	Quit Application
📈 Applications
Smart City Traffic Monitoring
Highway Traffic Analysis
Vehicle Counting Systems
Intelligent Transportation Systems
Traffic Research and Analytics
Computer Vision Learning Projects
🔮 Future Enhancements
Entry/Exit Vehicle Counting
Vehicle Speed Estimation
Red-Light Violation Detection
Wrong-Way Vehicle Detection
Traffic Dashboard using Streamlit
Real-Time Analytics Dashboard
Cloud Database Integration
Advanced Number Plate Detection Models
👩‍💻 Author

Developed as a Computer Vision and AI project using YOLOv8 and EasyOCR to demonstrate real-time traffic monitoring, analytics, tracking, and OCR capabilities.

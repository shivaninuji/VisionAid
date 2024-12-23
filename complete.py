import torch
from ultralytics import YOLO
import cv2
import numpy as np
import pyttsx3
import threading
from queue import Queue
from PIL import Image, ImageTk
import tkinter as tk
import time
import pytesseract
import matplotlib.pyplot as plt

# Set Tesseract path (adjust as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class IntegratedDetectionSystem:
    def __init__(self):  # Corrected constructor method
        print("Initializing Integrated Detection System...")
        
        # Initialize running flag
        self.running = True
        self.mode = "object_detection"  # Initial mode
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        print("YOLO model loaded successfully")
        
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Camera initialized")
        
        # Setup Tkinter window
        self.window = tk.Tk()
        self.window.title("Integrated Detection System")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create canvas for video
        self.canvas = tk.Canvas(self.window, width=1280, height=720)
        self.canvas.pack()
        
        # Camera parameters for object detection
        self.focal_length = 1000
        self.known_width = {
            'person': 0.5, 'chair': 0.5, 'bottle': 0.1,
            'laptop': 0.35, 'cell phone': 0.07, 'book': 0.15, 'cup': 0.08,
        }
        
        # Audio and detection settings
        self.audio_cooldown = 2
        self.last_audio_time = time.time()
        self.audio_queue = Queue()
        self.current_detections = []
        
        # Start audio thread
        self.audio_thread = threading.Thread(target=self.process_audio_queue, daemon=True)
        self.audio_thread.start()
        
        # Bind key events
        self.window.bind('<Key>', self.switch_mode)
        
        # OCR variables
        self.previous_text = ""
        
        print("System initialized and ready")
        self.speak("Integrated detection system activated")

    def switch_mode(self, event):
        """Switch between object detection and document reading modes"""
        if event.char == 'd':  # Press 'd' to switch modes
            self.mode = "object_detection" if self.mode == "document_reading" else "document_reading"
            mode_message = "Object Detection" if self.mode == "object_detection" else "Document Reading"
            self.speak(f"Switched to {mode_message} mode")

    def on_closing(self):
        """Handle window closing"""
        self.running = False
        self.window.destroy()

    def process_audio_queue(self):
        """Process audio messages from queue"""
        while self.running:
            if not self.audio_queue.empty():
                text = self.audio_queue.get()
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"Audio Error: {e}")
            time.sleep(0.1)

    def speak(self, text):
        """Add text to audio queue"""
        self.audio_queue.put(text)

    def process_ocr_frame(self, frame):
        """Process frame for OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # OCR configuration
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config).strip()

        return text, thresh

    def calculate_distance(self, pixel_width, actual_width):
        """Calculate distance using triangle similarity"""
        return (actual_width * self.focal_length) / pixel_width

    def get_position_description(self, object_center, frame_width):
        """Get position description"""
        center = frame_width / 2
        if object_center < center - frame_width/4:
            return "left"
        elif object_center > center + frame_width/4:
            return "right"
        return "middle"

    def process_frame(self):
        """Process a single frame based on current mode"""
        success, frame = self.camera.read()
        if not success:
            self.speak("Camera error")
            return False

        if self.mode == "object_detection":
            return self.process_object_detection_frame(frame)
        else:
            return self.process_document_reading_frame(frame)

    def process_object_detection_frame(self, frame):
        """Process frame for object detection"""
        results = self.model(frame, conf=0.5)
        current_time = time.time()
        
        self.current_detections = []
        height, width = frame.shape[:2]
        cv2.line(frame, (width//2, 0), (width//2, height), (150, 150, 150), 1, cv2.LINE_AA)
        
        for result in results[0]:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                conf = float(box.conf[0])
                
                if class_name in self.known_width:
                    pixel_width = x2 - x1
                    distance = self.calculate_distance(pixel_width, self.known_width[class_name])
                    
                    object_center = (x1 + x2) / 2
                    position = self.get_position_description(object_center, width)
                    
                    self.current_detections.append({
                        'class_name': class_name,
                        'distance': distance,
                        'position': position
                    })
                    
                    color = (0, 255, 0) if distance >= 2 else (0, 255, 255) if distance < 2 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Audio feedback
        if current_time - self.last_audio_time >= self.audio_cooldown and self.current_detections:
            self.current_detections.sort(key=lambda x: x['distance'])
            messages = [f"{detection['class_name']} {detection['distance']:.1f} meters {detection['position']}" 
                        for detection in self.current_detections[:3]]
            if messages:
                self.speak(". ".join(messages))
            self.last_audio_time = current_time
        
        return self.display_frame(frame)

    def process_document_reading_frame(self, frame):
        """Process frame for document reading"""
        text, processed_frame = self.process_ocr_frame(frame)
        
        # Convert processed frame for display
        processed_frame_color = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        
        if text and text != self.previous_text:
            print("Detected text:", text)
            self.speak(text)
            self.previous_text = text
        
        return self.display_frame(processed_frame_color)

    def display_frame(self, frame):
        """Display frame in Tkinter"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.img_tk = img_tk  # Keep a reference
        
        return True

    def run(self):
        """Main detection loop"""
        try:
            while self.running:
                if not self.process_frame():
                    break
                self.window.update()
                    
        except tk.TclError:  # Handle window closing
            pass
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.camera.isOpened():
            self.camera.release()
        self.engine.stop()
        print("System shutdown complete")

if __name__ == "__main__":
    detector = IntegratedDetectionSystem()
    detector.run()

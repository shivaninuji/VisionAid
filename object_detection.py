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

class ObjectDetectionAssistant:
    def __init__(self):
        print("Initializing Object Detection System...")
        
        # Initialize running flag first
        self.running = True
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        print("YOLO model loaded successfully")
        
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Adjust speaking rate
        self.engine.setProperty('volume', 1.0)  # Adjust volume
        
        # Initialize camera
        self.camera = cv2.VideoCapture(1)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Camera initialized")
        
        # Setup Tkinter window
        self.window = tk.Tk()
        self.window.title("Object Detection")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create canvas for video
        self.canvas = tk.Canvas(self.window, width=1280, height=720)
        self.canvas.pack()
        
        # Camera parameters
        self.focal_length = 1000
        self.known_width = {
            'person': 0.5,
            'chair': 0.5,
            'bottle': 0.1,
            'laptop': 0.35,
            'cell phone': 0.07,
            'book': 0.15,
            'cup': 0.08,
        }
        
        # Modified audio settings
        self.audio_cooldown = 2  # Keep consistent 2-second cooldown
        self.last_audio_time = time.time()  # Track last audio globally
        self.audio_queue = Queue()
        self.current_detections = []  # Store current frame detections
        
        # Start audio thread
        self.audio_thread = threading.Thread(target=self.process_audio_queue, daemon=True)
        self.audio_thread.start()
        
        print("System initialized and ready")
        self.speak("Object detection system activated")

    def on_closing(self):
        """Handle window closing"""
        self.running = False
        self.window.destroy()

    def calculate_distance(self, pixel_width, actual_width):
        """Calculate distance using triangle similarity"""
        return (actual_width * self.focal_length) / pixel_width

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

    def get_position_description(self, object_center, frame_width):
        """Get position description"""
        center = frame_width / 2
        if object_center < center - frame_width/4:
            return "left"
        elif object_center > center + frame_width/4:
            return "right"
        return "middle"

    def draw_detection_info(self, frame, box, class_name, conf, distance, position):
        """Draw detection information on frame"""
        x1, y1, x2, y2 = box
        
        # Calculate colors based on distance
        if distance < 1:
            color = (0, 0, 255)  # Red for close objects
        elif distance < 2:
            color = (0, 255, 255)  # Yellow for medium distance
        else:
            color = (0, 255, 0)  # Green for far objects
            
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create text with detection info
        text = f"{class_name} ({conf:.2f})"
        distance_text = f"{distance:.1f}m {position}"
        
        # Draw text
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, distance_text, (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def process_frame(self):
        """Process a single frame"""
        success, frame = self.camera.read()
        if not success:
            self.speak("Camera error")
            return False

        # Run detection
        results = self.model(frame, conf=0.5)
        current_time = time.time()
        
        # Clear current detections
        self.current_detections = []
        
        # Draw center line
        height, width = frame.shape[:2]
        cv2.line(frame, (width//2, 0), (width//2, height),
                (150, 150, 150), 1, cv2.LINE_AA)
        
        for result in results[0]:
            boxes = result.boxes
            for box in boxes:
                # Get detection info
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                conf = float(box.conf[0])
                
                if class_name in self.known_width:
                    # Calculate distance
                    pixel_width = x2 - x1
                    distance = self.calculate_distance(pixel_width, self.known_width[class_name])
                    
                    # Get position
                    object_center = (x1 + x2) / 2
                    position = self.get_position_description(object_center, width)
                    
                    # Store detection
                    self.current_detections.append({
                        'class_name': class_name,
                        'distance': distance,
                        'position': position
                    })
                    
                    # Draw detection info
                    self.draw_detection_info(frame, (x1, y1, x2, y2), 
                                          class_name, conf, distance, position)
        
        # Audio feedback every 2 seconds
        if current_time - self.last_audio_time >= self.audio_cooldown and self.current_detections:
            # Sort detections by distance (closest first)
            self.current_detections.sort(key=lambda x: x['distance'])
            
            # Create audio message for current frame
            messages = []
            for detection in self.current_detections[:3]:  # Limit to 3 closest objects
                distance_str = f"{detection['distance']:.1f} meters"
                messages.append(f"{detection['class_name']} {distance_str} {detection['position']}")
            
            if messages:
                self.speak(". ".join(messages))
            self.last_audio_time = current_time
        
        # Convert frame for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
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
    detector = ObjectDetectionAssistant()
    detector.run()
    from ultralytics.utils.benchmarks import benchmark


import cv2
import os
import numpy as np
import pytesseract
import pyttsx3
import time
from PIL import Image
from transformers import pipeline
import imutils
import datetime

# Configure Tesseract OCR (adjust path if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Constants for Distance Calculation
KNOWN_FACE_WIDTH = 14      # Average human face width in cm
FOCAL_LENGTH_FACE = 500    # Pre-calculated (can be calibrated)
FOCAL_LENGTH_OBJECT = 700  # Pre-calculated (can be calibrated)

# Menu options
MENU_OPTIONS = {
    "1": "Describe Scene",
    "2": "Read Text",
    "3": "Detect Faces",
    "4": "Navigate",
    "5": "Identify Objects",
    "6": "Time and Date",
    "7": "Add New Face",
    "8": "Exit"
}

# Load face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load object detection classifier
object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Global variables for known faces
known_face_encodings = []
known_face_names = []

# Initialize scene description model
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", framework="pt")

def load_known_faces(folder_path="known_faces"):
    """
    Loads images from the specified folder and trains the face recognizer
    """
    global known_face_encodings, known_face_names
    
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' not found. Creating folder.")
        os.makedirs(folder_path)
        print(f"Please add your known faces images (jpg files) into the '{folder_path}' folder.")
        return
    
    faces = []
    labels = []
    label_ids = {}
    current_id = 0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder_path, filename)
            name = os.path.splitext(filename)[0].capitalize()
            
            # Convert image to grayscale
            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect face in the image
            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in face_rects:
                face_roi = gray[y:y+h, x:x+w]
                faces.append(face_roi)
                
                if name not in label_ids:
                    label_ids[name] = current_id
                    current_id += 1
                
                labels.append(label_ids[name])
                known_face_names.append(name)
    
    if faces and labels:
        try:
            face_recognizer.train(faces, np.array(labels))
            print("Face recognizer trained successfully")
        except Exception as e:
            print(f"Error training face recognizer: {e}")
    else:
        print("No faces found in the known_faces directory")

def add_new_face():
    """
    Captures a new face image and adds it to the known faces directory
    """
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture the face, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Capture Face', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(faces) > 0:
                name = input("Enter the person's name: ")
                if name:
                    # Save the image
                    filename = f"known_faces/{name.lower()}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved face image as {filename}")
                    break
            else:
                print("No face detected in the frame. Please try again.")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def display_menu():
    """
    Displays the menu options
    """
    print("\n=== LIGHT_UP MENU ===")
    for key, value in MENU_OPTIONS.items():
        print(f"{key}. {value}")
    print("====================")

def speak_text(text):
    """
    Converts text to speech
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def describe_scene():
    """
    Captures an image and describes it via audio
    """
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture and describe scene, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Display the frame
        cv2.imshow('Scene Description', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Convert frame to RGB for the model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            try:
                # Get scene description
                result = captioner(pil_image)
                if result and isinstance(result, list) and "generated_text" in result[0]:
                    description = result[0]["generated_text"]
                    print(f"Scene: {description}")
                    speak_text(description)
                else:
                    print("No description available")
                    speak_text("No description available")
            except Exception as e:
                print(f"Error in scene description: {e}")
                speak_text("Error describing scene")
                
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def read_text():
    """
    Reads all text shown in live video
    """
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit text reading")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get better text detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Detect text
        text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
        
        # Display the frame with detected text
        cv2.putText(frame, "Detected Text:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Split text into lines and display
        y = 60
        for line in text.split('\n'):
            if line.strip():
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y += 20
        
        cv2.imshow('Text Reading', frame)
        
        # Speak the text if it's different from the last detected text
        if text:
            speak_text(text)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def detect_faces(frame):
    """
    Detects faces in the frame and displays a smooth video feed with face boxes and names
    """
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit face detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangle around faces and display name/distance
        for (x, y, w, h) in faces:
            # Calculate distance
            distance = calculate_distance(FOCAL_LENGTH_FACE, KNOWN_FACE_WIDTH, w)
            
            # Try to recognize the face
            face_roi = gray[y:y+h, x:x+w]
            try:
                label_id, confidence = face_recognizer.predict(face_roi)
                name = known_face_names[label_id] if confidence < 70 else "Unknown"
            except:
                name = "Unknown"
            
            # Draw rectangle with a thicker line
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add a filled background for text
            cv2.rectangle(frame, (x, y-30), (x+w, y), (0, 255, 0), -1)
            
            # Add text with better visibility
            cv2.putText(frame, f"{name} ({distance:.1f}cm)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def calculate_distance(focal_length, known_width, perceived_width):
    """
    Calculates the distance to an object using the focal length formula
    """
    if perceived_width > 0:
        return (known_width * focal_length) / perceived_width
    return -1

def navigate():
    """
    Provides real-time navigation guidance in live video
    """
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit navigation")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Divide frame into left, center, and right regions
        height, width = gray.shape
        left_region = gray[:, :width//3]
        center_region = gray[:, width//3:2*width//3]
        right_region = gray[:, 2*width//3:]
        
        # Calculate average brightness in each region
        left_brightness = np.mean(left_region)
        center_brightness = np.mean(center_region)
        right_brightness = np.mean(right_region)
        
        # Determine the clearest path
        if center_brightness > left_brightness and center_brightness > right_brightness:
            direction = "Continue straight"
            color = (0, 255, 0)  # Green
        elif left_brightness > right_brightness:
            direction = "Move slightly right"
            color = (0, 165, 255)  # Orange
        else:
            direction = "Move slightly left"
            color = (0, 0, 255)  # Red
        
        # Draw navigation overlay
        cv2.line(frame, (width//3, 0), (width//3, height), color, 2)
        cv2.line(frame, (2*width//3, 0), (2*width//3, height), color, 2)
        
        # Display guidance
        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add distance information
        cv2.putText(frame, f"Left: {left_brightness:.1f}", (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Center: {center_brightness:.1f}", (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Right: {right_brightness:.1f}", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Navigation', frame)
        
        # Speak the direction
        speak_text(direction)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def identify_objects():
    """
    Identifies objects shown in live video
    """
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit object detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect objects
        objects = object_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around objects
        for (x, y, w, h) in objects:
            # Calculate distance
            distance = calculate_distance(FOCAL_LENGTH_OBJECT, KNOWN_FACE_WIDTH, w)
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add label
            label = f"Object ({distance:.1f}cm)"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Speak the detection
            speak_text(f"Object detected at {distance:.1f} centimeters")
        
        # Display the frame
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to run the application
    """
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Load known faces
    load_known_faces()
    
    while True:
        display_menu()
        choice = input("Enter your choice (1-8): ")
        
        if choice == "8":
            print("Exiting...")
            break
            
        if choice == "1":
            describe_scene()
        elif choice == "2":
            read_text()
        elif choice == "3":
            detect_faces(frame)
        elif choice == "4":
            navigate()
        elif choice == "5":
            identify_objects()
        elif choice == "6":
            now = datetime.datetime.now()
            time_date_str = now.strftime("%I:%M %p, %A, %B %d, %Y")
            print(time_date_str)
            speak_text(time_date_str)
        elif choice == "7":
            add_new_face()
        else:
            print("Invalid choice. Please try again.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

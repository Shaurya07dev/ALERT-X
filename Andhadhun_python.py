import cv2
import os
import numpy as np
import pytesseract
import pyttsx3
import time
from PIL import Image
from transformers import pipeline
from deepface import DeepFace
import face_recognition
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
    "6": "Count People",
    "7": "Time and Date",
    "8": "Add New Face",
    "9": "Exit"
}

# Global variables for known faces
known_face_encodings = []
known_face_names = []

def load_known_faces(folder_path="known_faces"):
    """
    Loads images from the specified folder, computes face encodings,
    and extracts names from filenames.
    """
    global known_face_encodings, known_face_names
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' not found. Creating folder.")
        os.makedirs(folder_path)
        print(f"Please add your known faces images (jpg files) into the '{folder_path}' folder.")
        return
    
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0].capitalize()
                known_face_names.append(name)
                print(f"Loaded encoding for {name}")

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
            
        # Display the frame
        cv2.imshow('Capture Face', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            name = input("Enter the person's name: ")
            if name:
                # Save the image
                filename = f"known_faces/{name.lower()}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved face image as {filename}")
                
                # Reload known faces
                load_known_faces()
                break
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def display_menu():
    """
    Displays the menu options
    """
    print("\n=== ANDHADHUN MENU ===")
    for key, value in MENU_OPTIONS.items():
        print(f"{key}. {value}")
    print("=====================")

def speak_text(text):
    """
    Converts text to speech
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def describe_scene(frame):
    """
    Describes the current scene using AI
    """
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", framework="pt")
        result = captioner(pil_image)
        if result and isinstance(result, list) and "generated_text" in result[0]:
            description = result[0]["generated_text"]
            print(f"Scene: {description}")
            speak_text(description)
            return description
    except Exception as e:
        print(f"Error in scene description: {e}")
    return "No description available."

def detect_text(frame):
    """
    Detects and reads text from the frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6').strip()
    if text:
        print(f"Detected text: {text}")
        speak_text(text)
    else:
        print("No text detected")
        speak_text("No text detected")

def detect_faces(frame):
    """
    Detects and identifies known faces in the frame
    """
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        # Calculate distance
        face_width = right - left
        distance = calculate_distance(FOCAL_LENGTH_FACE, KNOWN_FACE_WIDTH, face_width)
        
        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({distance:.1f}cm)", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Speak the detection
        message = f"{name} detected at {distance:.1f} centimeters"
        print(message)
        speak_text(message)
    
    return frame

def calculate_distance(focal_length, known_width, perceived_width):
    """
    Calculates the distance to an object using the focal length formula
    """
    if perceived_width > 0:
        return (known_width * focal_length) / perceived_width
    return -1

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
        choice = input("Enter your choice (1-9): ")
        
        if choice == "9":
            print("Exiting...")
            break
            
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            continue
            
        if choice == "1":
            describe_scene(frame)
        elif choice == "2":
            detect_text(frame)
        elif choice == "3":
            frame = detect_faces(frame)
            cv2.imshow('Face Detection', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == "4":
            # Navigation logic here
            pass
        elif choice == "5":
            # Object detection logic here
            pass
        elif choice == "6":
            # People counting logic here
            pass
        elif choice == "7":
            now = datetime.datetime.now()
            time_date_str = now.strftime("%I:%M %p, %A, %B %d, %Y")
            print(time_date_str)
            speak_text(time_date_str)
        elif choice == "8":
            add_new_face()
        else:
            print("Invalid choice. Please try again.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

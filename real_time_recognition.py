import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load pre-trained CNN model
model = load_model(os.path.join('model', 'face_recognition_cnn.h5'))

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Class labels (assuming folder names as class labels)
class_labels = ['Person1', 'Person2', 'Person3']  # Modify based on your training data

def preprocess_face(face):
    face = cv2.resize(face, (128, 128))  # Resize to match model input
    face = face.astype("float") / 255.0  # Normalize pixel values
    face = img_to_array(face)  # Convert image to array
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        preprocessed_face = preprocess_face(face)
        
        prediction = model.predict(preprocessed_face)[0]
        label_index = np.argmax(prediction)
        label = class_labels[label_index]
        confidence = prediction[label_index] * 100
        
        # Draw rectangle around face and label it
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Face Recognition', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

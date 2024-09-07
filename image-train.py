import cv2
import os

# Set the name of the person
person_name = "person1"
output_dir = f"data/train/{person_name}"
os.makedirs(output_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

image_count = 0
while image_count < 50:  # Capture 50 images
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Capture Face', frame)
    
    # Save image
    img_path = os.path.join(output_dir, f"img{image_count}.jpg")
    cv2.imwrite(img_path, frame)
    
    image_count += 1
    
    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
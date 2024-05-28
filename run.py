import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

def visualize_facial_landmarks(image):
   """Detects facial landmarks in an image, visualizes them, and saves the output.

   Args:
       image: A NumPy array representing the image.

   Returns:
       None
   """

   mp_drawing = mp.solutions.drawing_utils
   mp_face_mesh = mp.solutions.face_mesh

   with mp_face_mesh.FaceMesh(
       max_num_faces=1,
       refine_landmarks=True,
       min_detection_confidence=0.5,
       min_tracking_confidence=0.5) as face_mesh:

       # Convert the image to RGB format (needed by MediaPipe)
       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

       # Process the image to detect facial landmarks
       results = face_mesh.process(image)

       # Draw the landmarks on the image, if any faces were detected
       if results.multi_face_landmarks:
           for face_landmarks in results.multi_face_landmarks:
               mp_drawing.draw_landmarks(
                   image=image,
                   landmark_list=face_landmarks,
                   connections=mp_face_mesh.FACEMESH_TESSELATION,
                   landmark_drawing_spec=None,
                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))

       # Convert the image back to BGR format before saving
       image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

       # Save the image with visualized landmarks
       cv2.imwrite('facial_landmarks_detection.jpg', image)




# Function to load an image
def load_image(image_path):
    return cv2.imread(image_path)

# Function to detect landmarks on the face
def detect_landmarks(image):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    result = face_detection.process(image)

    if result.detections:
        for detection in result.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            return x, y, w, h

    return None

# Function to calculate the height of the eyes
def calculate_eye_height(y, h):
    return int(h * 0.38)  # Adjust the multiplier based on your preferences

# Function to calculate the width of the glasses relative to the eyes
def calculate_glasses_width(w):
    return int(w * 0.9)  # Adjust the multiplier based on your preferences

# Function to place glasses based on eye landmarks
def place_glasses(image, x, y, w, h,gender,glass):
    glasses = cv2.imread(f"static/{glass}.png", -1)  # Replace "glasses.png" with your glasses image

    eye_height = calculate_eye_height(y, h)
    glasses_width = calculate_glasses_width(w)

    glasses = cv2.resize(glasses, (glasses_width, eye_height))
    alpha_glasses = glasses[:, :, 3] / 255.0
    alpha_face = 1.0 - alpha_glasses
    if(gender=='Man'):
       y_offset = y + eye_height //4 
    elif(gender=='Woman'):
       y_offset = y + eye_height //2   # Center glasses vertically on the eyes
    x_offset = x - (glasses_width - w) // 2  # Center glasses horizontally on the eyes

    roi = image[y_offset:y_offset + eye_height, x_offset:x_offset + glasses_width]

    for c in range(0, 3):
        image[y_offset:y_offset + eye_height, x_offset:x_offset + glasses_width, c] = (
            alpha_glasses * glasses[:, :, c] + alpha_face * roi[:, :, c])

    return image
def analyze_image(image):
    result=DeepFace.analyze(image,actions=["gender","age"])
    gender=result[0]['dominant_gender']
    age=result[0]['age']
    return gender,age
# Main function
def main(image,glass="glasses"):
    # image=load_image("face2.jpg")
    output_path = "static/output_image.jpg"  # Replace with the desired output path

    
    gender,age=analyze_image(image)
    if image is not None:
        landmarks = detect_landmarks(image)
        visualize_facial_landmarks(image)
        if landmarks:
            x, y, w, h = landmarks
            image = place_glasses(image, x, y, w, h,gender,glass)

            cv2.imwrite(output_path, image)
            # cv2.imshow("Output Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("I was here")
            # _, encoded_image = cv2.imencode('.png', image)

            return image
        else:
            print("No face detected in the image.")

    else:
        print("Failed to load the image.")



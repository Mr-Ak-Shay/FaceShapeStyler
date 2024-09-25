import cv2
import dlib
import numpy as np

predictor_path = '/Users/family/Downloads/shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

def get_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("No faces detected.")
        return None

    landmarks_list = []
    for face in faces:
        landmarks = face_predictor(gray, face)
        landmarks_np = np.zeros((68, 2), dtype="int")
        
        for i in range(68):
            landmarks_np[i] = (landmarks.part(i).x, landmarks.part(i).y)
        landmarks_list.append((landmarks_np, face))

    return landmarks_list

def classify_face_shape(landmarks):
    
    jaw_width = np.linalg.norm(landmarks[0] - landmarks[16])  
    face_height = np.linalg.norm(landmarks[8] - landmarks[27])  
    cheekbone_width = np.linalg.norm(landmarks[1] - landmarks[15])  

    print(f"Jaw Width: {jaw_width}, Face Height: {face_height}, Cheekbone Width: {cheekbone_width}")  

    if face_height == 0:  
        print("Face height is zero, unable to classify.")
        return "Unknown"

    ratio = jaw_width / face_height
    cheekbone_ratio = cheekbone_width / face_height

    print(f"Ratio: {ratio}, Cheekbone Ratio: {cheekbone_ratio}")  

    
    def determine_face_shape(cheekbone_ratio, ratio):
        if cheekbone_ratio > 1.0 and ratio > 1.1:
            return "Oval"
        elif cheekbone_ratio >= 1.5 and ratio < 1.5:
            return "Round"
        elif 1.3 <= cheekbone_ratio <= 1.5 and 1.4 <= ratio <= 1.6:
            return "Square"
        else:
            return "Rectangular"

    return determine_face_shape(cheekbone_ratio, ratio)

def recommend_hairstyle(face_shape):
    recommendations = {
        "Oval": "Medium length layered hairstyle",
        "Round": "High volume haircut to elongate face",
        "Square": "Soft, wispy layers or waves",
        "Rectangular": "Long layers to soften angular features",
        "Unknown": "Neutral style recommendation"
    }
    return recommendations.get(face_shape, "No recommendation available")

def detect_face_shape_and_recommend(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to read the image. Check the file path.")
        return
    
    landmarks_list = get_face_landmarks(image)
    
    if landmarks_list is None:
        return

    for landmarks_np, face in landmarks_list:
        
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

        face_shape = classify_face_shape(landmarks_np)
        hairstyle = recommend_hairstyle(face_shape)
        
        print(f"Detected Face Shape: {face_shape}")
        print(f"Recommended Hairstyle: {hairstyle}")

        for (x, y) in landmarks_np:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imshow("Face Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = "/Users/family/Downloads/circleface.jpg"
detect_face_shape_and_recommend(image_path)

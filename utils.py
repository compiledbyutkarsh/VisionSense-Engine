import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Initialize Webcam
cap = cv2.VideoCapture(0)

print("Starting video stream... Press 'q' to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error: Failed to access the camera.")
        break

    # Convert BGR to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    # Draw face detections
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw rectangle around face
            cv2.rectangle(image, bbox, (0, 255, 0), 3)

    cv2.imshow('FaceGuard AI - Detector', image)
    
    # Exit condition
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

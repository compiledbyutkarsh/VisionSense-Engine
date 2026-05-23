import cv2
import mediapipe as mp

# MediaPipe Face Detection setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

print("Camera chal raha hai... 'q' dabakar band karein.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Camera access nahi mil raha.")
        break

    # MediaPipe RGB image mangta hai
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    # Face detect hone par box banao
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            # Green box face ke charo taraf
            cv2.rectangle(image, bbox, (0, 255, 0), 3)

    cv2.imshow('Utkarsh ka MediaPipe Detector', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from deepface import DeepFace
import datetime

# Database folder jahan tumhari photo hogi (photo ka naam 'Utkarsh.jpg' hona chahiye)
db_path = "known_faces" 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    try:
        # Recognition + Emotion + Age/Gender detection
        results = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        
        for res in results:
            x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
            emotion = res['dominant_emotion']
            age = res['age']
            gender = res['dominant_gender']
            
            # Screen par details likho
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} | {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Attendance Log
            with open("attendance.txt", "a") as f:
                f.write(f"{datetime.datetime.now()} - Detected: {emotion}, {age}\n")
                
    except:
        pass

    cv2.imshow('Final AI Project', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
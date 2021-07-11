import cv2
from deepface import DeepFace

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    det = DeepFace.analyze(frame,['emotion'],enforce_detection= False)
    emotion = det['dominant_emotion']
    region = det['region']
    x = region['x']
    y = region['y']
    h = region['h']
    w = region['w']
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
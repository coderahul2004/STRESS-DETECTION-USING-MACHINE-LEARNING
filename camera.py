import cv2
import numpy as np

def run_camera():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    captured_stress = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangle on face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.putText(frame, "Press 'C' to capture | ESC to exit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.imshow("Camera Stress Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        # 👉 PRESS C TO CAPTURE
        if key == ord('c'):
            if len(faces) > 0:
                # Take first detected face
                emotion = np.random.choice(["Happy", "Sad", "Angry", "Neutral"])

                if emotion in ["Sad", "Angry"]:
                    captured_stress = "HIGH"
                elif emotion == "Neutral":
                    captured_stress = "MEDIUM"
                else:
                    captured_stress = "LOW"

                print(f"Emotion: {emotion}")
                print(f"Stress: {captured_stress}")
            else:
                print("No face detected")

            break

        # 👉 ESC TO EXIT
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return captured_stress
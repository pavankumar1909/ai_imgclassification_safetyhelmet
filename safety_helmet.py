import cv2
import mediapipe as mp
import numpy as np
import time

print("Mediapipe file:", mp.__file__)
print("Mediapipe version:", mp.__version__)

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# Eye landmark indices
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

cap = cv2.VideoCapture(0)

# PERCLOS variables
closed_frames = 0
total_frames = 0
start_time = time.time()
perclos = 0


def EAR(eye_points):

    A = np.linalg.norm(eye_points[1]-eye_points[5])
    B = np.linalg.norm(eye_points[2]-eye_points[4])
    C = np.linalg.norm(eye_points[0]-eye_points[3])

    ear = (A+B)/(2.0*C)

    return ear


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:

        cv2.putText(frame,"NO PERSON",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.imshow("Helmet Monitoring",frame)

        if cv2.waitKey(1)==27:
            break

        continue

    face_landmarks = results.multi_face_landmarks[0]

    h,w,_ = frame.shape

    left_eye = []
    right_eye = []

    for idx in LEFT_EYE:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        left_eye.append([x,y])

    for idx in RIGHT_EYE:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        right_eye.append([x,y])

    left_eye = np.array(left_eye)
    right_eye = np.array(right_eye)

    # Draw landmark dots
    for (x,y) in left_eye:
        cv2.circle(frame,(x,y),3,(255,0,0),-1)

    for (x,y) in right_eye:
        cv2.circle(frame,(x,y),3,(255,0,0),-1)

    # Draw eye contour
    cv2.polylines(frame,[left_eye],True,(0,255,255),1)
    cv2.polylines(frame,[right_eye],True,(0,255,255),1)

    left_ear = EAR(left_eye)
    right_ear = EAR(right_eye)

    ear = (left_ear + right_ear)/2

    total_frames += 1

    if ear < 0.21:
        closed_frames += 1
        state = "EYES CLOSED"
        color = (0,0,255)
    else:
        state = "EYES OPEN"
        color = (0,255,0)

    elapsed = time.time() - start_time

    if elapsed >= 30:

        perclos = (closed_frames/total_frames)*100
        print("PERCLOS:",perclos)

        closed_frames = 0
        total_frames = 0
        start_time = time.time()

    cv2.putText(frame,state,(40,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

    cv2.putText(frame,f"EAR: {ear:.2f}",(40,80),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

    cv2.putText(frame,f"PERCLOS: {perclos:.1f}%",(40,120),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    cv2.imshow("Helmet Monitoring",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()

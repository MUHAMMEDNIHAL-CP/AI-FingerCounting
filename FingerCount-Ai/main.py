import cv2 as cv
import mediapipe as mp

# Set up MediaPipe Hands and drawing utility
mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils

def getHandlandMark(frame, draw):
    lmlist = []
    
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

    frameRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    handsDetected = hands.process(frameRgb)
    if handsDetected.multi_hand_landmarks:
        for landmarks in handsDetected.multi_hand_landmarks:
            for id, lm in enumerate(landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append((id, cx, cy))
            if draw:
                mpDrawing.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)
    
    return lmlist

def fingerCount(lmlist):
    if len(lmlist) < 21:
        return 0

    count = 0
    if lmlist[8][2] < lmlist[6][2]:  # Index finger
        count += 1
    if lmlist[12][2] < lmlist[10][2]:  # Middle finger
        count += 1
    if lmlist[16][2] < lmlist[14][2]:  # Ring finger
        count += 1
    if lmlist[20][2] < lmlist[18][2]:  # Pinky finger
        count += 1
    if lmlist[4][1] < lmlist[2][1]:  # Thumb
        count += 1

    return count

# Start video capture
cam = cv.VideoCapture(0)

while True:
    success, frame = cam.read()

    if not success:
        print("Camera not detected!")
        continue

    lmlist = getHandlandMark(frame, draw=False)
    if lmlist:
        fc = fingerCount(lmlist=lmlist)
        cv.rectangle(frame, (400, 10), (600, 250), (0, 0, 0), -1)
        cv.putText(frame, str(fc), (400, 250), cv.FONT_HERSHEY_PLAIN, 20, (0, 255, 255), 30)

    cv.imshow("AI Finger Counting", frame)
    if cv.waitKey(1) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()

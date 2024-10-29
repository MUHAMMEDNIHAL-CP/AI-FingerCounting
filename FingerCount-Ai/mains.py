import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

finger_tips = [4, 8, 12, 16, 20]
thumb_tip = 5

def count_fingers(hand_landmarks):
    count = 0
    landmarks = hand_landmarks.landmark
    
    for tip_index in finger_tips[1:]: 
        if landmarks[tip_index].y < landmarks[tip_index - 2].y:
            count += 1
    
    if landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x:
        count += 1
    
    return count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            fingers_up = count_fingers(hand_landmarks)
            
            cv2.putText(frame, f'Fingers: {fingers_up}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (0.5)

def calculate_distance(hand_landmarks):
    x = [hand_landmarks.landmark[e].x for e in range(21)]
    y = [hand_landmarks.landmark[e].y for e in range(21)]
    width = max(x) - min(x)
    height = max(y) - min(y)
    distance = sum([euclidean_distance((x[0], y[0]), (x[e], y[e])) for e in [8, 12, 16, 20]])
    return distance / (width * height) ** (0.5)

def get_landmark(hand_landmarks, index):
    return (hand_landmarks.landmark[index].x, hand_landmarks.landmark[index].y)

def get_weighted_mean(images, weights):
    n, h, w, d = images.shape
    weights = weights[:n]
    result = np.zeros((h, w, d))
    for i in range(n):
        result += images[i] * weights[i]
    return result

cap = cv2.VideoCapture(1)
center = []

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
hold = False

all_frames = []

pyautogui.moveTo(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        all_frames.append(image)
        all_frames = all_frames[-3:]
        all_frames_array = np.array(all_frames)
        print(all_frames_array.shape)
        image = get_weighted_mean(all_frames_array, np.array([0.2, 0.3, 0.5])).astype(
            np.uint8
        )

        # image = all_frames_array[-1]
        print(image.shape)
        print(image)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        height, width, _ = image.shape
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (int(width*0.05), int(height*0.05)), (int(width*0.95), int(height*0.95)), (0, 0, 255), 2)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks[:1]:

                min_x = (
                    int(min([hand_landmarks.landmark[e].x for e in range(21)]) * width) - 20
                )
                max_x = (
                    int(max([hand_landmarks.landmark[e].x for e in range(21)]) * width) + 20
                )
                min_y = (
                    int(min([hand_landmarks.landmark[e].y for e in range(21)]) * height) - 20
                )
                max_y = (
                    int(max([hand_landmarks.landmark[e].y for e in range(21)]) * height) + 20
                )

                last_hold = hold
                dist = calculate_distance(hand_landmarks)
                hold = dist < 4.6

                COLOR = (0, 255, 0) if hold else (0, 0, 255)

                x, y = get_landmark(hand_landmarks, 0)
                center.append((int(x * width), int(y * height)))

                

                cv2.rectangle(image, (min_x, min_y), (max_x, max_y), COLOR, 2)
                cv2.circle(image, center[-1], 4, COLOR, 4)

                if len(center) >= 2:
                    center = center[-2:]
                    last_point = center[0]
                    current_point = center[1]
                    # print(center)
                    move_x = (last_point[0] - current_point[0]) * 3
                    move_y = (current_point[1] - last_point[1]) * 3
                    pyautogui.move(move_x, move_y)

                if hold and not last_hold:
                    pyautogui.mouseDown()
                elif not hold and last_hold:
                    pyautogui.mouseUp()
        

                # mp_drawing.draw_landmarks(
                #     image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style(),
                # )
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

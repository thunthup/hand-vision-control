import cv2
import mediapipe as mp
from util import getSectionFromXY,getMousePosFromSection, extractHandDistances,extractHandDistancesWithNoises
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import time
from scipy import stats
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import linear_model
import pyautogui
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
ratioList = []

labelList = []

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
gestureList = []

DIV = (4, 4)
CalibrateDIV = (5,5)
gestureClass = (0,1)
className = ("open fist", "close fist")
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

enableControl = 0
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
calibrating = 0
calibrateCounter = 0
fitted = 0
gClassifier = MLPClassifier(max_iter=500000,
                          hidden_layer_sizes=(15, ),random_state=1,early_stopping=True,n_iter_no_change=30,verbose=False)
xCursor = SCREEN_WIDTH//2
yCursor = SCREEN_HEIGHT//2
poly = PolynomialFeatures(degree=1)
scaler = StandardScaler()
last_hold = 0
def add_data(label):

    ratioList.append(extractedDists)
    labelList.append(label)
    


cv2.namedWindow("Calibrate Screen", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibrate Screen",
                      cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_hands=1,) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        blackScreen = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH))
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hands_landmarks in results.multi_hand_landmarks:
                hands_landmarks = hands_landmarks
                


                extractedDists = extractHandDistances(hands_landmarks) 
                if calibrating:
                    extractedDists = extractHandDistancesWithNoises(hands_landmarks,0.1)
                
                if fitted:
                    polyVariablesTemp = poly.transform(
                        [extractedDists])
                    scaledLiveData = scaler.transform(polyVariablesTemp)
                    gesture = gClassifier.predict(scaledLiveData)[0]
                    xHand= (hands_landmarks.landmark[0].x-0.4) * SCREEN_WIDTH*2.5
                    yHand = (hands_landmarks.landmark[0].y-0.4) * SCREEN_HEIGHT*2.5
                    xHand = max(xHand,0)
                    yHand = max(yHand,0)
                    xHand = int(min(xHand,SCREEN_WIDTH))
                    yHand = int(min(yHand,SCREEN_HEIGHT))
                    xCursor = int((0.9*xCursor) + (0.1 * xHand))
                    yCursor = int((0.9*yCursor) + (0.1 * yHand))
                    gestureList.append(gesture)
                    gestureList = gestureList[-3:]
                    #print(gestureList)
                    gesturePredict = int(round(np.mean(gestureList)))
                    
                    COLOR = (0, 255, 0) if gesturePredict == 1 else (0, 0, 255)
                    blackScreen = cv2.circle(
                        blackScreen, (xCursor, yCursor), 5, COLOR, 2)
                    blackScreen = cv2.putText(blackScreen, f'class: {gesturePredict}', (SCREEN_WIDTH//2, SCREEN_HEIGHT//2), cv2.FONT_HERSHEY_PLAIN,
                             2, (0, 255, 0), 3)
                    blackScreen = cv2.putText(blackScreen, f'gesture: {className[gesturePredict]}', (SCREEN_WIDTH//2, (SCREEN_HEIGHT//2)+50), cv2.FONT_HERSHEY_PLAIN,
                             2, (0, 255, 0), 3)
                    
                    if enableControl:
                        pyautogui.moveTo(xCursor, yCursor, duration=0)
                        if gesturePredict and not last_hold:
                            pyautogui.mouseDown()
                        elif not gesturePredict and last_hold:
                            pyautogui.mouseUp()
                        last_hold = gesturePredict    
                if not calibrating:
                    for i in range(DIV[0]):
                        for j in range(DIV[1]):
                            blackScreen = cv2.circle(
                                blackScreen, getMousePosFromSection((i, j), DIV, SCREEN_WIDTH , SCREEN_HEIGHT), 5, (0, 0, 0), 2)
                    
                if calibrating:
                    
                        if calibrateCounter < len(gestureClass):                           
                            label = gestureClass[calibrateCounter]
                            blackScreen = cv2.putText(blackScreen, f'class: {gestureClass[calibrateCounter]}', (SCREEN_WIDTH//2, SCREEN_HEIGHT//2), cv2.FONT_HERSHEY_PLAIN,
                             2, (0, 255, 0), 3)
                            blackScreen = cv2.putText(blackScreen, f'gesture: {className[calibrateCounter]}', (SCREEN_WIDTH//2, (SCREEN_HEIGHT//2)+50), cv2.FONT_HERSHEY_PLAIN,
                             2, (0, 255, 0), 3)
                           
                            cv2.imshow('Calibrate Screen', blackScreen)
                            if calibrateRep == 0:
                                time.sleep(1)
                            add_data(label)
                            

                            if calibrateRep < 150:
                                calibrateRep = calibrateRep + 1
                        
                            elif calibrateRep == 150:
                                calibrateRep = 0
                                blackScreen = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH))
                                cv2.imshow('Calibrate Screen', blackScreen)
                                
                            
                                if calibrateCounter < len(gestureClass) :
                                    calibrateCounter = calibrateCounter + 1
                                
                                              
                        else:
                            calibrating = 0
                            
                            polyVariables = poly.fit_transform(ratioList)
                            scaledData = scaler.fit_transform(polyVariables)
                            
                            gClassifier.fit(scaledData, labelList)
                            
                            print(gClassifier.score(scaledData, labelList))
                            
                            fitted = 1
                            
                    # Flip the image horizontally for a selfie-view display.
                    #     cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            mp_drawing.draw_landmarks(
                        image,
                        hands_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
        cv2.imshow('Calibrate Screen', blackScreen)
        cv2.imshow('MediaPipe Face Mesh', image)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            # print(mp_face_mesh.FACEMESH_IRISES)
            break

        
       
        elif k == ord('c'):
            calibrating = 1
            calibrateCounter = 0
            
            calibrateRep = 0
        elif k == ord('d'):
            enableControl = not enableControl

        elif k == ord('e'):
            
            dump(gClassifier, 'mlpClassifier.joblib') 
            dump(poly, 'polynomialFeatures.joblib') 
            dump(scaler, 'standardScaler.joblib') 
        elif k == ord('i'):
            fitted = 1
            gClassifier = load('mlpClassifier.joblib')
            poly = load('polynomialFeatures.joblib')
            scaler = load('standardScaler.joblib')

cap.release()
cv2.destroyAllWindows()

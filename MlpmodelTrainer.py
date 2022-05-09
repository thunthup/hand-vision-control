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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
ratioList = []
extractedDists = []
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
gClassifier = MLPClassifier(max_iter=500,
                          hidden_layer_sizes=(15, ),random_state=1,early_stopping=True,n_iter_no_change=30,verbose=False)
xCursor = SCREEN_WIDTH//2
yCursor = SCREEN_HEIGHT//2
poly = PolynomialFeatures(degree=1)
scaler = StandardScaler()
last_hold = 0
cap = None
def add_data(label,extractedDists):

    ratioList.append(extractedDists)
    labelList.append(label)
    
def trainFromVideo(videoName,real_label):
    cap = cv2.VideoCapture(videoName)
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_hands=1,) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

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
                    


                    
                    extractedDists = extractHandDistancesWithNoises(hands_landmarks,0.1)
                    
                    calibrateCounter = real_label                          
                    label = gestureClass[calibrateCounter]
                    blackScreen = cv2.putText(blackScreen, f'class: {gestureClass[calibrateCounter]}', (SCREEN_WIDTH//2, SCREEN_HEIGHT//2), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 3)
                    blackScreen = cv2.putText(blackScreen, f'gesture: {className[calibrateCounter]}', (SCREEN_WIDTH//2, (SCREEN_HEIGHT//2)+50), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 3)
                    
                    cv2.imshow('Calibrate Screen', blackScreen)
                    
                    add_data(label,extractedDists)
                                
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
        
    
def train_test_model():
    X_train, X_test, y_train, y_test = train_test_split(ratioList, labelList, test_size=0.3,
                                                    random_state=11)
    polyVariablesTrain = poly.fit_transform(X_train)
    scaledDataTrain = scaler.fit_transform(polyVariablesTrain)
    
    gClassifier.fit(scaledDataTrain, y_train)
    polyVariablesTemp = poly.transform(
                        X_test)
    scaledData = scaler.transform(polyVariablesTemp)
    
    predictions = gClassifier.predict(scaledData)
    cm = confusion_matrix(y_test, predictions, labels=gClassifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=gClassifier.classes_)
    disp.plot()

    plt.show()
    

cv2.namedWindow("Calibrate Screen", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibrate Screen",
                      cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


        
        
        
        
trainFromVideo('openMoveAll.mp4',0)    
trainFromVideo('closeMoveAll.mp4',1)      
train_test_model()
dump(gClassifier, 'mlpClassifierFromVideo.joblib') 
dump(poly, 'mlpPolynomialFeaturesFromVideo.joblib') 
dump(scaler, 'mlpStandardScalerFromVideo.joblib')
cv2.destroyAllWindows()

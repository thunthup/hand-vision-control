import math

import numpy as np


def calDist(lm1, lm2):
    # calculate the distance of (x1,y1,z1) and (x2,y2,z2) from landmarks
    return math.sqrt((lm1.x-lm2.x)**2+(lm1.y-lm2.y)**2+(lm1.z-lm2.z)**2)





def getSectionFromXY(X, Y, div, width=1920, height=1080):
    divx, divy = div
    xBorders = [(0 + width*e//divx) for e in range(divx)]
    yBorders = [(0 + height*e//divy) for e in range(divy)]
    xSection = 0
    ySection = 0
    for idx in range(divx-1, -1, -1):
        if X > xBorders[idx]:
            xSection = idx
            break
    for idx in range(divy-1, -1, -1):
        if Y > yBorders[idx]:
            ySection = idx
            break
    return (xSection, ySection)


def getMousePosFromSection(sections, div, width=1920, height=1080):
    divx, divy = div
    xSection, ySection = sections
    xPos = int(width*(xSection+0.5)/divx)
    yPos = int(height*(ySection+0.5)/divy)
    return (xPos, yPos)


def extractPoints(face_landmarks):
    extractedPoints = [[float(e.x), float(e.y), float(e.z)]
                       for e in face_landmarks.landmark[468:478]]
    pointsToExtract = [133, 33, 362, 263, 10,
                       152, 234, 454, 7, 163, 154, 155, 145]
    for point in pointsToExtract:
        extractedPoints.append([float(face_landmarks.landmark[point].x), float(
            face_landmarks.landmark[point].y), float(face_landmarks.landmark[point].z)])
    extractedPoints = np.array(extractedPoints)

    return extractedPoints.flatten()


def extractDistances(face_landmarks):
    landmark = face_landmarks.landmark
    leftIris = landmark[468]
    leftToExtract = [33,246,161,160,159,158,157,173,133,7,163,144,145,153,154,155]
    insideLeft = landmark[133]
    outsideLeft = landmark[33]
    leftEyeSize = calDist(insideLeft, outsideLeft)
    eyeData = []
    for i in leftToExtract:
        eyeData.append(calDist(leftIris,landmark[i])/leftEyeSize)
    # eyeData.append(calDist(landmark[161],landmark[163]))
    # eyeData.append(calDist(landmark[160],landmark[144]))
    #eyeData.append(calDist(landmark[159],landmark[145])/leftEyeSize)
    # eyeData.append(calDist(landmark[158],landmark[153]))
    # eyeData.append(calDist(landmark[157],landmark[154]))
    # eyeData.append(calDist(landmark[173],landmark[155]))
    insideRight = landmark[362]
    outsideRight = landmark[263]
    rightIris = landmark[473]
    rightEyeSize = calDist(insideRight, outsideRight)
    rightToExtract = [362,398,384,385,386,387,388,466,263,259,390,373,374,380,381,382]
    for i in rightToExtract:
        eyeData.append(calDist(rightIris,landmark[i])/rightEyeSize)
    
    return eyeData

def extractDistancesWithNoises(face_landmarks):
    landmark = face_landmarks.landmark
    leftIris = landmark[468]
    leftToExtract = [33,246,161,160,159,158,157,173,133,7,163,144,145,153,154,155]
    insideLeft = landmark[133]
    outsideLeft = landmark[33]
    leftEyeSize = calDist(insideLeft, outsideLeft)
    eyeData = []
    for i in leftToExtract:
        eyeData.append(calDist(leftIris,landmark[i])/leftEyeSize)
    # eyeData.append(calDist(landmark[161],landmark[163]))
    # eyeData.append(calDist(landmark[160],landmark[144]))
    #eyeData.append(calDist(landmark[159],landmark[145])/leftEyeSize)
    # eyeData.append(calDist(landmark[158],landmark[153]))
    # eyeData.append(calDist(landmark[157],landmark[154]))
    # eyeData.append(calDist(landmark[173],landmark[155]))
    insideRight = landmark[362]
    outsideRight = landmark[263]
    rightIris = landmark[473]
    rightEyeSize = calDist(insideRight, outsideRight)
    rightToExtract = [362,398,384,385,386,387,388,466,263,259,390,373,374,380,381,382]
    for i in rightToExtract:
        eyeData.append(calDist(rightIris,landmark[i])/rightEyeSize)
    noise = 1 + ((np.random.rand(32)-0.5) * 0.11)
    eyeData = np.array(eyeData)
    eyeData = eyeData * noise
    return eyeData


def extractDistances2(hands_landmarks):
    landmark = face_landmarks.landmark
    leftIris = landmark[468]
    leftToExtract = [113,225,224,223,222,221,189,244,233,232,231,230,229,228,31,226]
    insideLeft = landmark[133]
    outsideLeft = landmark[33]
    leftEyeSize = calDist(insideLeft, outsideLeft)
    eyeData = []
    for i in leftToExtract:
        eyeData.append(calDist(leftIris,landmark[i])/leftEyeSize)
    
    insideRight = landmark[362]
    outsideRight = landmark[263]
    rightIris = landmark[473]
    rightEyeSize = calDist(insideRight, outsideRight)
    rightToExtract = [446,342,445,444,443,442,441,413,464,453,452,451,450,449,448,261]
    for i in rightToExtract:
        eyeData.append(calDist(rightIris,landmark[i])/rightEyeSize)
    
    return eyeData

def extractHandDistances(hand_landmarks):
    landmark = hand_landmarks.landmark
    referencePoints = [0,1]
    pointsToExtract = [2,4,5,6,8,9,10,12,13,14,16,17,19,20]
    index_mcp = landmark[5]
    pinky_mcp = landmark[17]
    hand_size = calDist(index_mcp, pinky_mcp)
    handData = []
    for i in referencePoints:
        for j in pointsToExtract:
            handData.append(calDist(landmark[i],landmark[j])/hand_size)
    return handData

def extractHandDistancesWithNoises(hand_landmarks,noisePercent):
    handData = extractHandDistances(hand_landmarks)
    
    noise = 1 + ((np.random.rand(28)-0.5) * noisePercent)
    handData = np.array(handData)
    handData = handData * noise
    
    return handData

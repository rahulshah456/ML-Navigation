# -*- coding: utf-8 -*-

# ML Navigation Project
# cross-platform application to perform the basic user navigation
# with face movement and hand gestures for easy accessibility,
# by using the inbuilt webcam for input
#
# Created by: @rahulshah456
#


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5 import QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import imutils
from imutils import face_utils
from utils import *
import pyautogui as pag
import numpy as np
import dlib
import sys
import cv2


# Thresholds and consecutive frame length for triggering the mouse action.
MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSECUTIVE_FRAMES = 10
EYE_AR_THRESH = 0.19
EYE_AR_CONSECUTIVE_FRAMES = 10
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 10


# Initialize the frame counters for each action as well as
# booleans used to indicate if action is performed or not
MOUTH_COUNTER = 0
EYE_COUNTER = 0
WINK_COUNTER = 0
INPUT_MODE = False
EYE_CLICK = False
LEFT_WINK = False
RIGHT_WINK = False
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)


# Init Dlib's face detectors and predictors
shape_predictor = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)


# Initial the indexes of the facial landmarks for the left and
# right eye, nose and mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                # convert latest video frame to QImage format
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


                # Detect faces in the grayscale frame
                rects = detector(rgbImage, 0)

                # Loop over the face detections
                if len(rects) > 0:
                    rect = rects[0]

                    # Determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(frame, rect)
                    shape = face_utils.shape_to_np(shape)

                    # Extract the left and right eye coordinates, then use the
                    # coordinates to compute the eye aspect ratio for both eyes
                    mouth = shape[mStart:mEnd]
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    nose = shape[nStart:nEnd]

                    # Because I flipped the frame, left is right, right is left.
                    temp = leftEye
                    leftEye = rightEye
                    rightEye = temp

                    # Average the mouth aspect ratio together for both eyes
                    mar = mouth_aspect_ratio(mouth)
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    diff_ear = np.abs(leftEAR - rightEAR)

                    nose_point = (nose[3, 0], nose[3, 1])

                    # Compute the convex hull for the left and right eye, then
                    # visualize each of the eyes
                    mouthHull = cv2.convexHull(mouth)
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
                    cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

                    for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
                        cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)

                    # Check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink frame counter
                    if diff_ear > WINK_AR_DIFF_THRESH:

                        if leftEAR < rightEAR:
                            if leftEAR < EYE_AR_THRESH:
                                WINK_COUNTER += 1

                                if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                                    pag.click(button='left')

                                    WINK_COUNTER = 0

                        elif leftEAR > rightEAR:
                            if rightEAR < EYE_AR_THRESH:
                                WINK_COUNTER += 1

                                if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                                    pag.click(button='right')

                                    WINK_COUNTER = 0
                        else:
                            WINK_COUNTER = 0
                    else:
                        if ear <= EYE_AR_THRESH:
                            EYE_COUNTER += 1

                            if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                                SCROLL_MODE = not SCROLL_MODE
                                # INPUT_MODE = not INPUT_MODE
                                EYE_COUNTER = 0

                                # nose point to draw a bounding box around it

                        else:
                            EYE_COUNTER = 0
                            WINK_COUNTER = 0

                    if mar > MOUTH_AR_THRESH:
                        MOUTH_COUNTER += 1

                        if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
                            # if the alarm is not on, turn it on
                            INPUT_MODE = not INPUT_MODE
                            # SCROLL_MODE = not SCROLL_MODE
                            MOUTH_COUNTER = 0
                            ANCHOR_POINT = nose_point

                    else:
                        MOUTH_COUNTER = 0

                    # if INPUT_MODE:
                    #     cv2.putText(frame, "INPUT MODE", (10, 30),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE_COLOR, 2)
                    #     x, y = ANCHOR_POINT
                    #     nx, ny = nose_point
                    #     w, h = 60, 35
                    #     multiple = 1
                    #     cv2.rectangle(frame, (x - w, y - h),
                    #                 (x + w, y + h), GREEN_COLOR, 2)
                    #     cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

                    #     dir = direction(nose_point, ANCHOR_POINT, w, h)
                    #     cv2.putText(frame, dir.upper(), (10, 90),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE_COLOR, 2)
                    #     drag = 18
                    #     if dir == 'right':
                    #         pag.moveRel(drag, 0)
                    #     elif dir == 'left':
                    #         pag.moveRel(-drag, 0)
                    #     elif dir == 'up':
                    #         if SCROLL_MODE:
                    #             pag.scroll(40)
                    #         else:
                    #             pag.moveRel(0, -drag)
                    #     elif dir == 'down':
                    #         if SCROLL_MODE:
                    #             pag.scroll(-40)
                    #         else:
                    #             pag.moveRel(0, drag)

                    # if SCROLL_MODE:
                    #     cv2.putText(frame, 'SCROLL MODE IS ON!', (10, 60),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE_COLOR, 2)    


class Ui_MainWindow(QWidget):

    def setupUi(self, MainWindow):

        # home page of the main application
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(640, 480)
        MainWindow.setFocusPolicy(QtCore.Qt.StrongFocus)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame = QtWidgets.QFrame(MainWindow)
        self.frame.setGeometry(QtCore.QRect(-1, -1, 201, 481))
        self.frame.setAutoFillBackground(False)
        self.frame.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        # heading label for size navigation
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(10, 10, 171, 111))
        font = QtGui.QFont()
        font.setFamily("Microsoft Sans Serif")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 255, 255);"
                                 "margin:12px")
        self.label.setWordWrap(True)
        self.label.setObjectName("label")

        # side navigation bar labels
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 120, 160, 141))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.nav_face = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.nav_face.setStyleSheet("color: rgb(255, 255, 255);\n"
                                    "background-color: rgba(255, 255, 255,0.2);\n"
                                    "border-left: 3px solid rgba(255,255,255,1);"
                                    "padding-left: 12px\n")
        self.nav_face.setScaledContents(False)
        self.nav_face.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.nav_face.setWordWrap(True)
        self.nav_face.setObjectName("nav_face")
        self.verticalLayout.addWidget(self.nav_face)
        self.nav_fingers = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.nav_fingers.setStyleSheet("color: rgb(255, 255, 255);\n"
                                       "border-left: 3px solid rgba(255,255,255,0.5);\n"
                                       "padding-left: 12px\n")
        self.nav_fingers.setWordWrap(True)
        self.nav_fingers.setObjectName("nav_fingers")
        self.verticalLayout.addWidget(self.nav_fingers)
        self.nav_gestures = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.nav_gestures.setStyleSheet("color: rgb(255, 255, 255);\n"
                                        "border-left: 3px solid rgba(255,255,255,0.5);"
                                        "padding-left: 12px\n")
        self.nav_gestures.setWordWrap(True)
        self.nav_gestures.setObjectName("nav_gestures")
        self.verticalLayout.addWidget(self.nav_gestures)
        self.nav_voice = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.nav_voice.setStyleSheet("color: rgb(255, 255, 255);\n"
                                     "border-left: 3px solid rgba(255,255,255,0.5);"
                                     "padding-left: 12px\n")
        self.nav_voice.setWordWrap(True)
        self.nav_voice.setObjectName("nav_voice")
        self.verticalLayout.addWidget(self.nav_voice)

        # settings button
        self.settingsButton = QtWidgets.QPushButton(self.frame)
        self.settingsButton.setEnabled(True)
        self.settingsButton.setGeometry(QtCore.QRect(20, 420, 151, 31))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.settingsButton.sizePolicy().hasHeightForWidth())
        self.settingsButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Microsoft Sans Serif")
        font.setBold(True)
        font.setWeight(75)
        self.settingsButton.setFont(font)
        self.settingsButton.setMouseTracking(False)
        self.settingsButton.setStyleSheet("color: rgb(255, 255, 255);\n"
                                          "border: 0.5px solid white;")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./res/settings.svg"),
                       QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.settingsButton.setIcon(icon)
        self.settingsButton.setIconSize(QtCore.QSize(32, 16))
        self.settingsButton.setAutoDefault(True)
        self.settingsButton.setDefault(False)
        self.settingsButton.setFlat(True)
        self.settingsButton.setObjectName("settingsButton")

        # video frame for face navigation
        self.videoFrame = QtWidgets.QLabel(MainWindow)
        self.videoFrame.setGeometry(QtCore.QRect(210, 10, 421, 231))
        self.videoFrame.setStyleSheet("border: 1px solid rgba(0,0,0, 0.5);")
        self.videoFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.videoFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.videoFrame.setObjectName("videoFrame")

        # checkboxes menu to activate and deactivate features
        self.label_2 = QtWidgets.QLabel(MainWindow)
        self.label_2.setGeometry(QtCore.QRect(220, 260, 201, 16))
        font = QtGui.QFont()
        font.setFamily("Microsoft Sans Serif")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(MainWindow)
        self.verticalLayoutWidget_2.setGeometry(
            QtCore.QRect(220, 280, 171, 131))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Microsoft Sans Serif")
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout_2.addWidget(self.checkBox)
        self.checkBox_2 = QtWidgets.QCheckBox(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Microsoft Sans Serif")
        self.checkBox_2.setFont(font)
        self.checkBox_2.setObjectName("checkBox_2")
        self.verticalLayout_2.addWidget(self.checkBox_2)
        self.checkBox_3 = QtWidgets.QCheckBox(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Microsoft Sans Serif")
        self.checkBox_3.setFont(font)
        self.checkBox_3.setObjectName("checkBox_3")
        self.verticalLayout_2.addWidget(self.checkBox_3)
        self.checkBox_4 = QtWidgets.QCheckBox(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Microsoft Sans Serif")
        self.checkBox_4.setFont(font)
        self.checkBox_4.setObjectName("checkBox_4")
        self.verticalLayout_2.addWidget(self.checkBox_4)

        # face navigation window buttons - detect, activate
        self.horizontalLayoutWidget = QtWidgets.QWidget(MainWindow)
        self.horizontalLayoutWidget.setGeometry(
            QtCore.QRect(390, 430, 231, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(
            self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.saveButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.saveButton_2.setEnabled(True)
        self.saveButton_2.setMouseTracking(False)
        self.saveButton_2.setStyleSheet("background-color: rgba(0,0,0,0.5);\n"
                                        "color: rgb(255, 255, 255);\n")
        self.saveButton_2.setAutoDefault(True)
        self.saveButton_2.setDefault(False)
        self.saveButton_2.setFlat(False)
        self.saveButton_2.setObjectName("saveButton_2")
        self.horizontalLayout.addWidget(self.saveButton_2)
        self.saveButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.saveButton.setEnabled(True)
        self.saveButton.setMouseTracking(False)
        self.saveButton.setStyleSheet("background-color: rgb(85, 170, 255);\n"
                                      "color: rgb(255, 255, 255);\n")
        self.saveButton.setAutoDefault(True)
        self.saveButton.setDefault(False)
        self.saveButton.setFlat(False)
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout.addWidget(self.saveButton)

        #subscribe the threads
        th = Thread(MainWindow)
        th.changePixmap.connect(self.setImage)
        th.start()

        # add widgets texts with translation support
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ML Navigation"))
        logo = QtGui.QIcon()
        logo.addPixmap(QtGui.QPixmap("./res/logo.svg"),
                       QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(logo)
        self.label.setText(_translate(
            "MainWindow", "SELECT YOUR NAVIGTION OPTION"))
        self.nav_face.setText(_translate("MainWindow", "Face Navigation"))
        self.nav_fingers.setText(_translate("MainWindow", "Hand Fingers"))
        self.nav_gestures.setText(_translate("MainWindow", "Hand Gestures"))
        self.nav_voice.setText(_translate("MainWindow", "Voice Navigation"))
        self.settingsButton.setText(_translate("MainWindow", "Settings"))
        self.label_2.setText(_translate(
            "MainWindow", "Check to enable features"))
        self.checkBox.setText(_translate(
            "MainWindow", "Left Eye Blink to LeftClick"))
        self.checkBox_2.setText(_translate(
            "MainWindow", "Right Eye Blink to RightClick"))
        self.checkBox_3.setText(_translate(
            "MainWindow", "Move Mouse Pointer with face"))
        self.checkBox_4.setText(_translate("MainWindow", "Mouth Open to Undo"))
        self.saveButton_2.setText(_translate("MainWindow", "Detect Camera"))
        self.saveButton.setText(_translate("MainWindow", "Enable / Save"))

    @pyqtSlot(QImage)
    def setImage(self, image):
        # update video frame as image to the window
        self.videoFrame.setPixmap(QPixmap.fromImage(image))
            


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QDialog()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from question_one import *
from question_two import *
from question_three import *
from question_four import *
from functools import partial

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        global text 
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(925, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(50, 60, 191, 441))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift")
        font.setBold(False)
        font.setWeight(50)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(30, 50, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Pristina")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton{\n"
"    border-radius: 20px;\n"
"    background-color:\n"
"    rgb(85,175,255);\n"
"    color:rgb(0, 0, 127);\n"
"}\n"
"QPushButton:hover{\n"
"    background-color:rgb(153, 238, 255);\n"
"}\n"
"QPushButton:pressed{\n"
"    \n"
"    background-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5, stop:0 rgba(92, 197, 205, 255), stop:1 rgba(255, 255, 255, 255));\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(findCorner)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 110, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Pristina")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(findIntrinsic)
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 170, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Pristina")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("")
        self.pushButton_3.setObjectName("pushButton_3")


        self.selectImageText = QtWidgets.QLineEdit(self.groupBox)
        self.selectImageText.setGeometry(QtCore.QRect(50, 250, 101, 31))
        self.selectImageText.setObjectName("selectImageText")
        #self.selectImageText.setAcceptRichText(False) for QTextEdit

        self.pushButton_3.clicked.connect(partial(findExtrinsic, self.selectImageText))

        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 300, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Pristina")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setStyleSheet("")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(findDistortion)
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 370, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Pristina")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setStyleSheet("")
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(findUndistortion)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(40, 220, 100, 21))
        self.label.setObjectName("label")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(260, 60, 191, 441))
        self.groupBox_2.setObjectName("groupBox_2")
        self.textEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.textEdit.setGeometry(QtCore.QRect(20, 60, 141, 41))
        self.textEdit.setObjectName("textEdit")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_6.setGeometry(QtCore.QRect(10, 160, 171, 41))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.clicked.connect(partial(augment_word, self.textEdit, True))
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_7.setGeometry(QtCore.QRect(10, 230, 171, 41))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.clicked.connect(partial(augment_word, self.textEdit, False))
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(470, 60, 191, 441))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_8.setGeometry(QtCore.QRect(10, 160, 171, 41))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_8.clicked.connect(disparityMap)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(680, 60, 191, 441))
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_9.setGeometry(QtCore.QRect(10, 90, 171, 41))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.clicked.connect(find_keypoints)
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_10.setGeometry(QtCore.QRect(10, 180, 171, 41))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_10.clicked.connect(draw_matchedKeypoints)
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_11.setGeometry(QtCore.QRect(10, 270, 171, 41))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_11.clicked.connect(warp_image)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 925, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Calibration"))
        self.pushButton.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.pushButton_2.setText(_translate("MainWindow", "1.2 Find Intrinsic"))
        self.pushButton_3.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.pushButton_4.setText(_translate("MainWindow", "1.4 Find Distortion"))
        self.pushButton_5.setText(_translate("MainWindow", "1.5 Show Result"))
        self.label.setText(_translate("MainWindow", "Select image:"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Augmented Reality"))
        self.pushButton_6.setText(_translate("MainWindow", "2.1 Show Words on Board"))
        self.pushButton_7.setText(_translate("MainWindow", "2.2 Show Words Vertically"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3 Stereo Disparity Map"))
        self.pushButton_8.setText(_translate("MainWindow", "3.1 Stereo Disparity Map"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. SIFT"))
        self.pushButton_9.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.pushButton_10.setText(_translate("MainWindow", "4.2 Matched keypoints"))
        self.pushButton_11.setText(_translate("MainWindow", "4.3 Warp Image"))

    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
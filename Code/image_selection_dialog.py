# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Luis\Documents\rvau2.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

class ImageDlg(QtWidgets.QDialog):

    def __init__(self):
        super(ImageDlg, self).__init__()
        self.setupUi(self)
        self.images = []
        self.imageName = set()
        self.name = None


    def setupUi(self, Dialog):

        Dialog.setObjectName("Dialog")
        Dialog.resize(809, 599)
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 811, 601))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(100, 20, 100, 20)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setMinimumSize(QtCore.QSize(200, 0))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setKerning(True)
        self.label_2.setFont(font)
        self.label_2.setAutoFillBackground(True)
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.textEdit = QtWidgets.QTextEdit(self.verticalLayoutWidget)
        self.textEdit.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textEdit.setFont(font)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout_3.addWidget(self.textEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(50, 50, 50, 50)
        self.horizontalLayout_2.setSpacing(50)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.listWidget = QtWidgets.QListWidget(self.verticalLayoutWidget)
        self.listWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.listWidget.setAlternatingRowColors(False)
        self.listWidget.setObjectName("listWidget")
        self.horizontalLayout_2.addWidget(self.listWidget)
        self.horizontalLayout_2.setStretch(0, 3)
        self.horizontalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(50, -1, 50, 10)
        self.horizontalLayout.setSpacing(50)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.AddImage = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.AddImage.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.AddImage.setFont(font)
        self.AddImage.setObjectName("AddImage")
        self.horizontalLayout.addWidget(self.AddImage)
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.deleteImage = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.deleteImage.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.deleteImage.setFont(font)
        self.deleteImage.setObjectName("deleteImage")
        self.horizontalLayout.addWidget(self.deleteImage)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 5)
        self.verticalLayout.setStretch(2, 1)

        self.retranslateUi(Dialog)
        self.listWidget.setCurrentRow(-1)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.pushButton.clicked.connect(self.finish)
        self.AddImage.clicked.connect(self.addImage)
        self.deleteImage.clicked.connect(self.delete)




    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_2.setText(_translate("Dialog", "Interest Point Name:"))
        self.AddImage.setText(_translate("Dialog", "Add Image"))
        self.pushButton.setText(_translate("Dialog", "Finish"))
        self.deleteImage.setText(_translate("Dialog", "Delete Last Image"))


    def addImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if fileName and fileName not in self.imageName:
            img = cv2.imread(fileName)

            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio )
            self.label.setPixmap(pixmap)
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            array = fileName.split('/')
            self.listWidget.addItem(array [len(array)-1 ])
            self.imageName.add(fileName)

            self.images.append(img)

    def delete(self):
        #self.label.setPixmap(None)
        size = self.listWidget.count()
        itemP = self.listWidget.item(size-1)
        self.listWidget.takeItem(size-1)
        self.imageName.pop()
        self.images.pop()

    def finish(self):
        self.name = self.textEdit.toPlainText()
        if not self.name or len(self.images) < 1:
            info_box = QtWidgets.QMessageBox(self)
            info_box.setIcon(QtWidgets.QMessageBox.Critical)
            info_box.setText("Name can't be empty and you must pick at least one image!")
            return info_box.exec()
        else:
            self.accept()



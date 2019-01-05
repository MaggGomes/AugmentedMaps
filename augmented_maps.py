import os

import cv2
import numpy as np
import keyboard
from PyQt5 import (QtWidgets as qt,
                   QtGui as gui,
                   QtCore as qtc)
from PyQt5.QtCore import Qt

import utils
from interest_point_augment_graphic import InterestPointAugmentGraphic
from image_map import ImageMap
from database import Database
from preparation import Preparation


class AugmentedMaps(qt.QMainWindow):
    def __init__(self):
        super().__init__()
        self.database: Database = None
        self.database = Database.connect('db.db')
        self.configure_window()
        self.configure_menu()
        self.__entryWindow = None
        self.scene = qt.QGraphicsScene()
        self.view = qt.QGraphicsView(self.scene)
        self.popup_list: EntriesList = None
        self.setCentralWidget(self.view)
        self.show()

    def configure_menu(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        file_menu = menubar.addMenu('Augmentation')

        open_act = qt.QAction('Augment Map', self)
        open_act.triggered.connect(self.open_image_map)
        file_menu.addAction(open_act)

        capture_video = qt.QAction('Capture Video', self)
        capture_video.triggered.connect(self.open_capture)
        file_menu.addAction(capture_video)

        exit_action = qt.QAction('Quit', self)
        exit_action.triggered.connect(qt.qApp.quit)
        file_menu.addAction(exit_action)

        menubar.addAction(file_menu.menuAction())

        database_menu = menubar.addMenu('Preparation')

        add_database_action = qt.QAction('Add Map', self)
        add_database_action.triggered.connect(self.open_add_entry_window)
        database_menu.addAction(add_database_action)

        list_entries_act = qt.QAction('List Maps', self)
        list_entries_act.triggered.connect(self.list_entries)
        database_menu.addAction(list_entries_act)

        menubar.addAction(database_menu.menuAction())

    def configure_window(self):
        self.setWindowTitle('Augmented Maps')
        screen_size = gui.QGuiApplication.primaryScreen().availableSize()
        self.resize(int(screen_size.width() * 3 / 5),
                    int(screen_size.height() * 3 / 5))
        self.center()
        self.statusBar().showMessage('Ready')

    def open_capture(self):
        self.scene.clear()
        a = 0
        kp = None
        goodImages = []
        found_match = False
        counter = 0
        video = cv2.VideoCapture(0)

        while True:
            a = a + 1
            check, frame = video.read()

            if check:
                # In order to reduce computer power:
                if (not found_match and counter % 5 == 0) or (found_match and counter % 10 == 0):
                    found_match, kp, _, goodImages = self.compute_match(
                        frame, database)
                if found_match:
                    frame = self.augment_map(
                        kp, goodImages[0][0], frame, goodImages[0][1])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    counter = counter + 1
                except:
                    counter = 0
                if not found_match:
                    # Get width and height from image
                    h, w, __ = np.shape(frame)
                    # Draws a circle at the center of the map
                    frame = utils.draw_center_map(frame, w, h)

                # show frame
                self.scene.addPixmap(gui.QPixmap(utils.numpy_to_qimage(frame)))

                # cv2.imshow('image', frame)
                # key = cv2.waitKey(1)
                if keyboard.is_pressed('q'):
                    break

        video.release()

        self.scene.clear()

    def open_image_map(self):
        filename, __ = qt.QFileDialog.getOpenFileName(self, 'Load Image', os.environ.get('HOME'),
                                                      'Images (*.jpg *.jpeg *.png)')
        if filename:
            image = cv2.imread(filename)
            img = image
            found, kp, img, goodImages = self.compute_match(
                image, self.database)
            if True == found:
                image = self.augment_map(
                    kp, goodImages[0][0], img, goodImages[0][1])
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Draw result in screen
            self.scene.clear()
            self.scene.addPixmap(gui.QPixmap(utils.numpy_to_qimage(image)))
            self.update()

    @staticmethod
    def compute_match(image, database):
        image_hist_eq = utils.histogram_equalization(image)
        kp, des = utils.get_features(image_hist_eq)

        goodImages = []

        for entry in database.entries:
            # print(f"Matching features with {entry.name}")
            matches = utils.match_descriptors(entry.descriptors, des)
            # print(f"Found {len(matches)} descriptor matches")

            if len(matches) >= 50:
                print(f"Found a match: {entry.name}")
                goodImages.append((matches, entry))
                goodImages.append((matches, entry))

        if len(goodImages) == 0:
            return False, None, None, []

        # Sorts images according to the number of matches
        goodImages = sorted(goodImages, key=lambda x: len(x[0]))

        # Augments map
        return True, kp, image, goodImages

    @staticmethod
    def augment_map(kp, matches, image, image_prepared):
        # Calculates source and destination points
        src_pts = np.float32([image_prepared.keypoints[m.queryIdx]['pt']
                              for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Get width and height from image
        h, w, __ = np.shape(image)

        # Converts color namespace from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Verifies if the image map has any Point of Interest
        if len(image_prepared.interestPoints) > 0:
            # Gets the nearest Point of Interest from the center
            nearest_interestpoint = utils.get_nearest_interestpoint(
                image_prepared, matrix, w, h)

            # Resize the image of the Point of Interest
            interestImage = cv2.cvtColor(
                nearest_interestpoint[3], cv2.COLOR_BGR2RGB)
            if image.shape[0] > image.shape[1]:
                interestPointImage = cv2.resize(
                    interestImage, (int(0.30*image.shape[1]), int(0.25*image.shape[0])), interpolation=cv2.INTER_CUBIC)
            elif image.shape[0] <= image.shape[1]:
                interestPointImage = cv2.resize(
                    interestImage, (int(0.25*image.shape[1]), int(0.30*image.shape[0])), interpolation=cv2.INTER_CUBIC)
            else:
                interestPointImage = cv2.resize(
                    interestImage, (int(0.30*image.shape[1]), int(0.30*image.shape[0])), interpolation=cv2.INTER_CUBIC)

            # Calculates the centroid of the Point of Interest image to be drawn
            interestPointCentroid = utils.get_centroid(
                (nearest_interestpoint[0][0][0], nearest_interestpoint[0][1][0], nearest_interestpoint[0][2][0], nearest_interestpoint[0][3][0]))

            # Verifies the location of the Point of Interest and calculates the position of the its image associated to be drawn
            if interestPointCentroid[0] < w/2:
                interesPointImageXi = w - interestPointImage.shape[1]
                interesPointImageYi = h - interestPointImage.shape[0]
                interesPointImageXf = w
                interesPointImageYf = h
                interestPointImageCorderX = interesPointImageXi
                interestPointImageCorderY = interesPointImageYi - 29
            else:
                interesPointImageXi = 0
                interesPointImageYi = h - interestPointImage.shape[0]
                interesPointImageXf = interestPointImage.shape[1]
                interesPointImageYf = h
                interestPointImageCorderX = interesPointImageXf
                interestPointImageCorderY = interesPointImageYi - 29

            # Draw image of the Point of Interest in the map
            image[interesPointImageYi:interesPointImageYf,
                  interesPointImageXi: interesPointImageXf] = interestPointImage

            # Draws an header for the Point of Interest Image
            headerPts = utils.get_header_points(
                interesPointImageXi, interesPointImageYi, interesPointImageXf)

            image = cv2.fillPoly(
                image, [np.int32(headerPts)], (255, 255, 255))

            # Draws a line from the header to the center of the nearest Point of Interest
            cv2.line(image, (int(interestPointCentroid[0]), int(interestPointCentroid[1])), (
                int(interestPointImageCorderX), int(interestPointImageCorderY)), (255, 255, 255), 2)

            # Calculates distance between the center and the Point of Interest
            scale = image_prepared.scale
            interestPointDistance = int(scale * nearest_interestpoint[1])

            interestPointText = nearest_interestpoint[2] + \
                " - " + str(interestPointDistance) + " m"

            # Draw name of the Point of Interest
            cv2.putText(image, interestPointText, (
                int(headerPts[1][0][0] + 5), int(headerPts[1][0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0)

            # Draws the location of the nearest Point of Interest
            image = cv2.polylines(
                image, [np.int32(nearest_interestpoint[0])], True, 255, 3, cv2.LINE_AA)

        # Gets the points of the compass
        pts_compass = utils.get_compass_points(w, h)

        # Project corners into frame
        dst_compass = cv2.perspectiveTransform(pts_compass, matrix)

        wDiff = int(w/2 + 30 - dst_compass[0][0][0])
        hDiff = int(h/2 - dst_compass[0][0][1])

        dst_compass[0][0][0] = dst_compass[0][0][0] + wDiff
        dst_compass[0][0][1] = dst_compass[0][0][1] + hDiff
        dst_compass[1][0][0] = dst_compass[1][0][0] + wDiff
        dst_compass[1][0][1] = dst_compass[1][0][1] + hDiff
        dst_compass[2][0][0] = dst_compass[2][0][0] + wDiff
        dst_compass[2][0][1] = dst_compass[2][0][1] + hDiff
        dst_compass[3][0][0] = dst_compass[3][0][0] + wDiff
        dst_compass[3][0][1] = dst_compass[3][0][1] + hDiff

        # Connect the corners of the compass with lines
        image = cv2.polylines(
            image, [np.int32(dst_compass)], True, 0, 2, cv2.LINE_AA)

        image = cv2.fillPoly(
            image, [np.int32([dst_compass[0], dst_compass[2], dst_compass[3]])], (150, 0, 0))

        image = cv2.fillPoly(
            image, [np.int32([dst_compass[0], dst_compass[1], dst_compass[2]])], (0, 0, 150))

        # Draws a circle at the center of the map
        image = utils.draw_center_map(image, w, h)

        return image

    def open_add_entry_window(self):
        print('Opening an image map')
        self.__entryWindow = Preparation(self.database)
        pos = self.frameGeometry().topLeft()
        self.__entryWindow.move(pos.x() + 20, pos.y() + 20)
        self.__entryWindow.show()

    def list_entries(self):
        self.popup_list = EntriesList(self, self.database)

    def center(self):
        qr = self.frameGeometry()
        cp = qt.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        reply = qt.QMessageBox.question(self, 'Message',
                                        "Are you sure to you want to quit?", qt.QMessageBox.Yes |
                                        qt.QMessageBox.No, qt.QMessageBox.No)

        if reply == qt.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


class EntriesList(qt.QWidget):
    class ListEntry(qt.QWidget):
        deleted = qtc.pyqtSignal(qt.QWidget)

        def __init__(self, parent, entry: ImageMap):
            super().__init__(parent)
            self.entry = entry
            layout = qt.QGridLayout()
            image = qt.QLabel()
            image.setPixmap(gui.QPixmap(
                utils.image_to_qimage(entry.img)).scaledToWidth(300))
            layout.addWidget(image, 0, 0, Qt.AlignCenter)
            layout.addWidget(qt.QLabel("%s" %
                                       (entry.name)), 1, 0, Qt.AlignCenter)
            delete_btn = qt.QPushButton("Delete", self)
            delete_btn.released.connect(lambda: self.deleted.emit(self))
            layout.addWidget(delete_btn, 2, 0, Qt.AlignCenter)
            self.setLayout(layout)

    def __init__(self, parent, database):
        super().__init__(parent)
        self.database = database

        self.area = qt.QScrollArea()
        widget = qt.QWidget()
        self.layout = qt.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        for e in self.database.entries:
            list_entry = self.ListEntry(self, e)
            list_entry.deleted.connect(self.delete_entry)
            self.layout.addWidget(list_entry)

        widget.setLayout(self.layout)
        self.area.setWidget(widget)
        self.area.show()

    def delete_entry(self, entry: ListEntry):
        self.database.remove_map(entry.entry)
        self.layout.removeWidget(entry)
        entry.deleteLater()
        self.layout.update()

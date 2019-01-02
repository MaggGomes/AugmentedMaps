import os
import cv2
from typing import Optional, Tuple
from PyQt5 import (QtWidgets as qt,
                   QtGui as gui,
                   QtCore as qtc)
from PyQt5.QtCore import Qt
from image_map import ImageMap
from database import Database
from gui_editor import EditorScene, EditorState, EditorView
import utils


class Preparation(qt.QMainWindow):
    entry_saved = qtc.pyqtSignal(ImageMap)

    def __init__(self, database: Database):
        super().__init__()
        self._database = database
        self.img = None
        self.toolbar: qt.QToolBar = None
        self.tool_save: qt.QAction = None
        self.configure_window()
        self.configure_toolbar()

        self.editor_scene = EditorScene()
        self.editor_scene.entry_changed.connect(self.on_entry_change)
        self.editor_view = EditorView(self.editor_scene)
        self.editor_view.mouse_moved.connect(self.on_mouse_move)
        self.entry_name_combo: qt.QComboBox = None
        self.sidebar = self.create_sidebar()

        splitter = qt.QSplitter(Qt.Horizontal, self)

        splitter.addWidget(self.editor_view)
        splitter.addWidget(self.sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.setCentralWidget(splitter)

    def create_sidebar(self) -> qt.QWidget:
        sidebar = qt.QWidget()
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        area = qt.QScrollArea()
        area.setFrameStyle(0)
        content = qt.QWidget()
        content_layout = qt.QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 8)

        info_box = qt.QWidget()
        form = qt.QFormLayout()
        self.entry_name_combo = qt.QComboBox(info_box)
        self.entry_name_combo.setEditable(True)
        self.entry_name_combo.setEditText('')

        info_box.setLayout(form)
        content_layout.addWidget(info_box)

        self.augments_group = qt.QButtonGroup(self)
        self.augments_group.setExclusive(False)
        self.augments_group.buttonClicked[int].connect(self.augment_clicked)

        augments_widget = qt.QWidget(sidebar)
        augments_layout = qt.QGridLayout()

        box_augment_widget, box_augment_button = self.toolbox_interestpoint()
        self.augments_group.addButton(
            box_augment_button, 1)
        augments_layout.addWidget(box_augment_widget, 0, 0)

        augments_widget.setLayout(augments_layout)

        toolbox = qt.QToolBox(sidebar)
        toolbox.addItem(augments_widget, "Points of Interest")
        content_layout.addWidget(toolbox)

        content_layout.setSizeConstraint(qt.QLayout.SetMinimumSize)
        content.setLayout(content_layout)
        area.setWidget(content)
        layout.addWidget(area)
        sidebar.setLayout(layout)
        return sidebar

    # Configures window where the image map will be rendered
    def configure_window(self):
        self.setWindowTitle('New Map')
        screen_size = gui.QGuiApplication.primaryScreen().availableSize()
        self.resize(int(screen_size.width() * 3 / 5),
                    int(screen_size.height() * 3 / 5))
        self.grabGesture(qtc.Qt.PinchGesture)
        self.statusBar().showMessage("Load a map to start")

    # Configures toolbar
    def configure_toolbar(self):
        self.toolbar = self.addToolBar('Main Toolbar')

        load_act = self.toolbar_button('Load Image')
        load_act.triggered.connect(self.load_image)
        self.toolbar.addAction(load_act)

        self.tool_save = self.toolbar_button('Save Map')
        self.tool_save.setDisabled(True)
        self.tool_save.triggered.connect(self.save_map)
        self.toolbar.addAction(self.tool_save)

    # Loads a image map
    def load_image(self):
        filename, __ = qt.QFileDialog.getOpenFileName(self, 'Load Image', os.environ.get('HOME'),
                                                      'Images (*.jpg *.jpeg *.png)')
        if filename:
            print(f'Loading {filename}')
            self.img = cv2.imread(filename)
            self.editor_scene.load_map(self.img)

    # Saves the prepared map in database
    def save_map(self):
        name = self.entry_name_combo.currentText()

        # Map prepared must have a name
        if not name:
            info_box = qt.QMessageBox(self)
            info_box.setIcon(qt.QMessageBox.Critical)
            info_box.setText("Name can't be empty!")
            return info_box.exec()

        # Computes keypoints and descriptors of the image map
        sift = cv2.xfeatures2d.SIFT_create()
        image_hist_eq = utils.histogram_equalization(self.img)
        kp, des = sift.detectAndCompute(image_hist_eq, None)

        # Because pickling cv2.KeyPoint causes PicklingError, we need to create a new abstraction for it
        keypoints = utils.keypoints_to_kpdict(kp)

        interestPoints = [a.getInterestPoint()
                          for a in self.editor_scene.augments]

        # Creates a new ImageMap with the data from the manipulated image
        imageMap = ImageMap(name, self.editor_scene.entry['img'], keypoints, des,
                            interestPoints)
        self._database.add_map(imageMap)

        info_box = qt.QMessageBox(self)
        info_box.setIcon(qt.QMessageBox.Information)
        info_box.setText("Saved successfully as '%s'" % imageMap.name)
        info_box.exec()
        self.entry_saved.emit(imageMap)

    def toolbar_button(self, text: str) -> qt.QAction:
        action = qt.QAction(text, self)
        return action

    def toolbox_interestpoint(self) -> Tuple[qt.QWidget, qt.QToolButton]:
        button = qt.QToolButton()
        button.setText("Mark point of interest")
        button.setCheckable(True)
        button.setMinimumSize(60, 60)
        grid = qt.QGridLayout()
        grid.addWidget(button, 0, 0, Qt.AlignCenter)
        widget = qt.QWidget()
        widget.setLayout(grid)
        return widget, button

    def augment_clicked(self, id: int):
        clicked = self.augments_group.button(id)
        for button in self.augments_group.buttons():
            if clicked != button:
                button.setChecked(False)

        if clicked.isChecked():
            print("Creating a Point of Interest")
            self.editor_scene.state = EditorState.INSERT_AUGMENT_ITEM
        else:
            self.editor_scene.state = EditorState.NONE

    def on_entry_change(self):
        self.tool_save.setDisabled(self.editor_scene.entry is None)

    def on_mouse_move(self, scene_position: Tuple[float, float]):
        if self.editor_scene.entry is None:
            return
        status = '(x: {:d}, y: {:d})'.format(
            int(scene_position[0]), int(scene_position[1]))
        self.statusBar().showMessage(status)

    def closeEvent(self, event):
        reply = qt.QMessageBox.question(self, 'Message',
                                        "You haven't saved the entry yet!<br>"
                                        "Are you sure you want to close?", qt.QMessageBox.Yes |
                                        qt.QMessageBox.No, qt.QMessageBox.No)
        if reply == qt.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

from enum import Enum, unique, auto
from typing import List, Set, Tuple

from PyQt5 import (QtWidgets as qt,
                   QtGui as gui,
                   QtCore as qtc)
from PyQt5.QtCore import Qt
import numpy as np
import math
import cv2
from interest_point_augment_graphic import InterestPointAugmentGraphic
from image_selection_dialog import ImageDlg

@unique
class EditorState(Enum):
    NONE = auto()
    INSERT_AUGMENT_ITEM = auto()


class EditorScene(qt.QGraphicsScene):
    entry_changed = qtc.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.state: EditorState = EditorState.NONE
        self.entry: dict = None
        self._selection_rect: dict = None
        self._selection_rect_ui: qt.QGraphicsRectItem = None

        self.augments: Set[InterestPointAugmentGraphic] = set()
        self._dragging: InterestPointAugmentGraphic = None
        self._selected: InterestPointAugmentGraphic = None
        self._item_start_point: Tuple[float, float] = None
        self._item: InterestPointAugmentGraphic = None

        self.delete_act = qt.QAction('Delete Point of Interest', self)
        self.delete_act.triggered.connect(self.delete_point_of_interest)

    def load_map(self, img):
        self.state = EditorState.NONE
        self._selected = None
        self._dragging = None
        self.clear()

        # Get map dimensions
        h, w, d = np.shape(img)

        # Converts map to RGB
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        q_image = gui.QImage(rgbImg, w, h, w * d, gui.QImage.Format_RGB888)
        entry_item = self.addPixmap(gui.QPixmap(q_image))
        self.setSceneRect(entry_item.boundingRect())
        self.update()
        self.entry = {'id': None,
                      'img': img,
                      'gui': entry_item}
        self.entry_changed.emit()

    def delete_point_of_interest(self):
        if self._selected:
            self.augments.remove(self._selected)
            self.removeItem(self._selected)
            self._selected = None
            self.update()

    def contextMenuEvent(self, event: qt.QGraphicsSceneContextMenuEvent):
        item = self.itemAt(event.scenePos(), gui.QTransform())
        item = item if item and isinstance(
            item, InterestPointAugmentGraphic) else self._selected
        if item:
            self._selected = item
            context_menu = qt.QMenu()
            context_menu.addAction(self.delete_act)
            context_menu.exec(event.screenPos())
            event.accept()
        else:
            event.ignore()

    def mousePressEvent(self, event: qt.QGraphicsSceneMouseEvent):
        if event.button() != Qt.LeftButton:
            return

        pos = event.scenePos()

        if self.state is EditorState.INSERT_AUGMENT_ITEM:
            self._item_start_point = (pos.x(), pos.y())
        elif self.state is EditorState.NONE:
            item = self.itemAt(event.scenePos(), gui.QTransform())
            item = item if item and isinstance(
                item, InterestPointAugmentGraphic) else None
            self._dragging = item
            if self._selected and item != self._selected:
                self._selected.selected = False
            self._selected = item
            if self._selected:
                self._selected.selected = True
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event: qt.QGraphicsSceneMouseEvent):
        if self.state is EditorState.INSERT_AUGMENT_ITEM:
            if self._item:
                self._item.drawing = False
                dlg = ImageDlg()
                if dlg.exec():
                    name = dlg.name
                    images = dlg.images
                    self._item.setName(name)
                    self._item.setImages(images)
                else:
                    self._selected = self._item
                    self.delete_point_of_interest()

                    return
                self.augments.add(self._item)
                self._item_start_point = None
                self._item = None
                self.update()
        elif self.state is EditorState.NONE:
            if self._dragging:
                self._dragging.dragging = False
                self._dragging = None
                self.update()

        event.accept()

    def mouseMoveEvent(self, event: qt.QGraphicsSceneMouseEvent):
        if self.state is EditorState.INSERT_AUGMENT_ITEM:
            if self._item_start_point:
                if not self._item:
                    self._item = InterestPointAugmentGraphic(0, 0)
                    self._item.setPos(*self._item_start_point)
                    self._item.drawing = True
                    self.addItem(self._item)
                box = self._item
                box.width = event.scenePos().x() - self._item_start_point[0]
                box.height = event.scenePos().y() - self._item_start_point[1]
                self.update()
        elif self.state is EditorState.NONE and self._dragging is not None:
            curr = (event.scenePos().x(), event.scenePos().y())
            prev = (event.lastScenePos().x(), event.lastScenePos().y())
            delta = (curr[0] - prev[0], curr[1] - prev[1])
            self._dragging.dragging = True
            self._dragging.moveBy(*delta)
            self.update()
        event.accept()


class EditorView(qt.QGraphicsView):
    mouse_moved = qtc.pyqtSignal(object)

    def __init__(self, scene: EditorScene):
        super().__init__(scene)
        self.editor_scene = scene
        scene.entry_changed.connect(self.handle_entry_changed)
        self.viewport().grabGesture(Qt.PinchGesture)
        self.viewport().setMouseTracking(True)
        self.setFrameStyle(0)

    def handle_entry_changed(self):
        self.reset_zoom()
        self.fit_to_entry()

    def fit_to_entry(self):
        if self.editor_scene.entry is not None:
            self.fitInView(self.editor_scene.entry['gui'], Qt.KeepAspectRatio)

    def reset_zoom(self):
        self.resetTransform()

    def resizeEvent(self, event: gui.QResizeEvent):
        self.fit_to_entry()
        super().resizeEvent(event)

    def viewportEvent(self, event: qtc.QEvent):
        if event.type() == qtc.QEvent.Gesture:
            return self.gesture_event(event)
        return super().viewportEvent(event)

    def mouseMoveEvent(self, event: gui.QMouseEvent):
        scene_pos = self.mapToScene(event.pos())
        self.mouse_moved.emit((scene_pos.x(), scene_pos.y()))
        super().mouseMoveEvent(event)

    def gesture_event(self, event: qt.QGestureEvent) -> bool:
        pinch: qt.QPinchGesture = event.gesture(Qt.PinchGesture)
        if pinch is not None:
            zoom_factor = pinch.totalScaleFactor()
            self.setTransformationAnchor(qt.QGraphicsView.NoAnchor)
            self.setResizeAnchor(qt.QGraphicsView.NoAnchor)
            self.scale(zoom_factor, zoom_factor)
        return True

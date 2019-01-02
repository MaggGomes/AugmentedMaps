from abc import ABC
from typing import Optional
from PyQt5 import (QtWidgets as qt,
                   QtGui as gui,
                   QtCore as qtc)
from PyQt5.QtCore import Qt
import math
from interest_point import InterestPoint


class InterestPointAugmentGraphic(qt.QGraphicsItem):
    def __init__(self, width: float = 100, height: float = 100):
        super().__init__()
        self._dragging: bool = False
        self._selected: bool = False
        self._drawing: bool = False
        self.setCursor(Qt.PointingHandCursor)
        self._width: float = width
        self._height: float = height
        self._name = None
        self._image = None

    @property
    def dragging(self) -> bool:
        return self._dragging

    @dragging.setter
    def dragging(self, value: bool):
        self.setCursor(Qt.ClosedHandCursor if value else Qt.PointingHandCursor)
        self._dragging = value

    @property
    def selected(self) -> bool:
        return self._selected

    @selected.setter
    def selected(self, value: bool):
        self._selected = value

    @property
    def drawing(self) -> bool:
        return self._drawing

    @drawing.setter
    def drawing(self, value: bool):
        self._drawing = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width: float):
        self.prepareGeometryChange()
        self._width = width

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height: float):
        self.prepareGeometryChange()
        self._height = height

    def boundingRect(self) -> qtc.QRectF:
        return qtc.QRectF(0, 0, self.width + 5, self.height + 5)

    def paint(self, painter: gui.QPainter, option: qt.QStyleOptionGraphicsItem, widget: Optional[qt.QWidget] = ...):
        color = gui.QColor(255, 0, 0, 255)
        if self.dragging:
            color.setAlphaF(0.7)
        if self._selected:
            color.setGreen(200)
        if self._drawing:
            color = gui.QColor(120, 32, 32, 100)
        pen = gui.QPen()
        pen.setColor(color)
        pen.setWidth(5)
        painter.setPen(pen)
        painter.drawRect(2, 2, int(self.width), int(self.height))

    def getInterestPoint(self):
        interestPoint = InterestPoint(
            self.x(), self.y(), self.width, self.height)
        interestPoint.setName(self._name)
        interestPoint.setImage(self._image)

        return interestPoint

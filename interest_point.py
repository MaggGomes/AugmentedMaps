# Class that abstracts a point of interest in the map
class InterestPoint():
    def __init__(self, x, y, w, h):
        super().__init__()
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.name = None
        self.images = set()

    def setName(self, name):
        self.name = name

    def setImage(self, image):
        self.images = image

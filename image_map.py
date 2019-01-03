import pickle
from typing import List, Optional
import numpy as np
from interest_point import InterestPoint


class ImageMap:
    def __init__(self, name: str, scale, image, keypoints, descriptors, interestPoints: Optional[List[InterestPoint]] = None):
        self.name = name
        self.img = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.interestPoints = interestPoints if interestPoints is not None else []
        self.scale = scale
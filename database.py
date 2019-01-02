import pickle
from typing import List, Optional
import numpy as np
from image_map import ImageMap


# Abstraction to save/return image maps prepared
class Database:
    def __init__(self, filename):
        self.filename = filename
        self.__imageMaps = dict()

    @classmethod
    def connect(cls, filename):
        try:
            file = open(filename, 'rb')
        except FileNotFoundError:
            db = cls(filename)
            db.save()
            return db
        else:
            return pickle.load(file)

    # Returns the saved maps
    @property
    def entries(self):
        return self.__imageMaps.values()

    # Returns a map
    def imageMap(self, name) -> Optional[ImageMap]:
        return self.__imageMaps.get(name)

    # Saves the current state of the database
    def save(self):
        with open(self.filename, 'wb') as file:
            print(f'Saving data to database: {self.filename}')
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    # Adds a new map to the database
    def add_map(self, imageMap: ImageMap):
        self.__imageMaps[imageMap.name] = imageMap
        self.save()

    # Removes a map from the database
    def remove_map(self, imageMap: ImageMap):
        if self.__imageMaps[imageMap.name]:
            del self.__imageMaps[imageMap.name]
            self.save()

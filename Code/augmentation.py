import cv2
from augmented_maps import AugmentedMaps
import utils
import numpy as np
from database import Database


def captureVideo(database):
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
                found_match, kp, _, goodImages = AugmentedMaps.compute_match(
                    frame, database)
            if found_match:
                frame = AugmentedMaps.augment_map(
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
            cv2.imshow('image', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    database = Database.connect('db.db')
    captureVideo(database)

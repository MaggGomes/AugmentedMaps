import cv2


def captureVideo():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Augmentation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    captureVideo()

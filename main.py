import sys
from PyQt5 import QtWidgets
from augmented_maps import AugmentedMaps


def main():
    print ('Argument List: ' + str(sys.argv))
    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) > 1:
        window = AugmentedMaps(True)
    else:
        window = AugmentedMaps(False)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

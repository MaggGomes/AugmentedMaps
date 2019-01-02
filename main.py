import sys
from PyQt5 import QtWidgets
from augmented_maps import AugmentedMaps

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AugmentedMaps()
    sys.exit(app.exec_())

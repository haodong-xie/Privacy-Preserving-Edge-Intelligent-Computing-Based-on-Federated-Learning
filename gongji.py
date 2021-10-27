
from PyQt5.QtWidgets import QApplication,QDialog,QFileDialog
from PyQt5.QtGui import QPixmap
import time
import sys

from device_gongji.Dev_Gongji import Ui_Dialog as Anquan

class Window_gongji(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.gongji_ui = Anquan()
        self.gongji_ui.setupUi(self)
        self.btn = self.gongji_ui.jiazia
        self.yuanshi = self.gongji_ui.yuanshi
        self.jieguo = self.gongji_ui.gongji
        self.btn.clicked.connect(self.jiazai)

    def jiazai(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择数据', 'F:\\Data\\attack', 'data(*.json)')
        time.sleep(1)
        # self.label.setPixmap(QPixmap(fname))
        self.yuanshi.setPixmap(QPixmap('F:\\Data\\test\\3.png'))
        time.sleep(5)
        self.jieguo.setPixmap(QPixmap("F:\\Data\\test\\80.png"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    G = Window_gongji()
    G.show()
    sys.exit(app.exec_())
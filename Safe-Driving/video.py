#!/usr/bin/env python

import nmap  # import nmap.py module
import sys
import vlc
import os

from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from ui import *

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.qList = []  # 存储ip列表
        # creating a basic vlc instance
        self.instance = vlc.Instance()
        # creating an empty vlc media player
        self.mplayer = self.instance.media_player_new()

        self.listView.clicked.connect(self.clickedlist)  # listview 的点击事件
        self.ButScanIP.clicked.connect(self.ScanIP_do)
        self.ButOpenCamera.clicked.connect(self.openCamera)  # 打开相机显示图像

    def openCamera(self, ):
        """
        打开相机开始显示
        :return:
        """

        media = self.instance.media_new("http://192.168.0.103:8555")
        media.get_mrl()
        self.mplayer.set_media(media)
        self.mplayer.set_hwnd(self.playWidget.winId())  # 设定Qt窗体的控件的 句柄为显示窗口
        self.mplayer.play()  # 开始显示

    def PlayPause(self):
        if self.mediaplayer.is_playing():
            return
        else:
            if self.mediaplayer.play() == -1:
                self.openCamera()
                return
            self.mediaplayer.play()

    def updateUI(self):

        self.positionslider.setValue(self.mediaplayer.get_position() * 1000)
        if not self.mediaplayer.is_playing():
            return

    def clickedlist(self, qModelIndex):
        QMessageBox.information(self, "QListView", "你选择了: " + self.qList[qModelIndex.row()])
        print("点击的是：" + str(qModelIndex.row()))

    def ScanIP_do(self):
        nm = nmap.PortScanner()  # instantiate nmap.PortScanner object
        nm.scan(hosts='192.168.0.0/24', arguments='-n -sP -PE -PA21,23,80,3389')
        hosts_list = [(x, nm[x]['status']['state']) for x in nm.all_hosts()]
        slm = QStringListModel();  # 创建mode

        self.qList.clear()  # 重置清空列表

        for host, status in hosts_list:
            self.qList.append('{0}:{1}'.format(host, status))
            print('{0}:{1}'.format(host, status))
        slm.setStringList(self.qList)  # 将数据设置到model
        self.listView.setModel(slm)  ##绑定 listView 和 model


if __name__ == '__main__':
    app = QApplication(sys.argv)

    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())









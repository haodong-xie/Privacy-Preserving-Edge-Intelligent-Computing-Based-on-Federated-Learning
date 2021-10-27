import codecs
import json
import pickle
import warnings
from argparse import ArgumentParser
from flask import *
from flask_socketio import *
from PyQt5.QtWidgets import QApplication,QMainWindow,QDialog,QFileDialog
import sys
from PyQt5.QtCore import QThread,pyqtSignal
from ser_welcome.Ser_welcome import Ui_MainWindow
from ser_xunlian.ser_xunlian import Ui_Dialog as Second
from ser_dengluguanli.ser_denglu import Ui_Dialog as Denglu
from ser_xitongrizhi.ser_xitongrizhi import Ui_Dialog as Rizhi
from ser_gongji.Ser_gongji import Ui_Dialog as Gongji
from ser_welcome.Ser_canshu import Ui_Dialog as Canshu
from models.Fed import FedAvg
from utils.weight_operation import *
import time
import datetime
warnings.filterwarnings("ignore")
global yonghuming
yonghuming = ''
class Window_First(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)
        self.denglu_btn = self.main_ui.dengluguanli
        self.xunlian_btn = self.main_ui.xunlian
        self.rihzi_btn = self.main_ui.rizhi_2

    def work(self):
        self.thread = Server_thread()
        self.thread.start()
        print("进程启动成功")

class Window_Canshu(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.canshu_ui = Canshu()
        self.canshu_ui.setupUi(self)
        self.btn = self.canshu_ui.kaishi
        self.btn.clicked.connect(First.work)

    def get_para(self):
        self.zongshu = self.canshu_ui.yonghuzongshu.text()
        self.yuzhi = self.canshu_ui.yuzhi.text()
        self.lunshu = self.canshu_ui.lunshu.text()
        return self.zongshu,self.yuzhi,self.lunshu

class Server_thread(QThread):
    trigger = pyqtSignal()
    def __init__(self, parent=None):
        super(Server_thread, self).__init__()
    def run(self):
        host = "127.0.0.1"
        port = 10099
        n,k,lun = CanShu.get_para()
        parser = ArgumentParser()
        Cargs = parser.parse_args()
        with open('./init_file/commandline_args.json', 'r') as f:
            Cargs.__dict__ = json.load(f)
        drop = 1
        server = secaggserver(host, port, int(n), int(k), drop, Cargs,int(lun))
        server.start()

class Window_Second(QDialog):
    txt_signal = pyqtSignal(int,str)
    def __init__(self):
        QDialog.__init__(self)
        self.ser_train_ = Second()
        self.ser_train_.setupUi(self)
        self.txt_signal[int,str].connect(self.printf_2)
        self.ser_tongxun = self.ser_train_.tongxin_2
        self.ser_juhe = self.ser_train_.juhe
        self.dis_ = [self.ser_tongxun,self.ser_juhe]

    def printf(self,int1, str1):
        self.txt_signal[int,str].emit(int1,str1)
    def printf_2(self,int, mypstr):
        self.dis = self.dis_[int]
        self.dis.append(mypstr)
        self.cursor = self.dis.textCursor()
        self.dis.moveCursor(self.cursor.End)
        QApplication.processEvents()

    def printf_1(self):
        self.dis_1.append(str(self.info_data["0"]))
        self.cursor = self.dis_1.textCursor()
        self.dis_1.moveCursor(self.cursor.End)
        QApplication.processEvents()
        self.dis_2.append(str(self.info_data["1"]))
        self.cursor = self.dis_2.textCursor()
        self.dis_2.moveCursor(self.cursor.End)
        QApplication.processEvents()
        self.dis_3.append(str(self.info_data["2"]))
        self.cursor = self.dis_3.textCursor()
        self.dis_3.moveCursor(self.cursor.End)
        QApplication.processEvents()
        self.dis_4.append(str(self.info_data["3"]))
        self.cursor = self.dis_4.textCursor()
        self.dis_4.moveCursor(self.cursor.End)
        QApplication.processEvents()

class Window_Dengluguanli(QDialog):
    txt_signal = pyqtSignal(str)
    def __init__(self):
        QDialog.__init__(self)
        self.ser_dengluguanli = Denglu()
        self.ser_dengluguanli.setupUi(self)
        self.txt_signal[str].connect(self.printf_2)
        self.shuchu = self.ser_dengluguanli.shuchu

    def printf(self, str1):
        self.txt_signal[str].emit(str1)
    def printf_2(self, mypstr):
        self.dis = self.shuchu
        self.dis.append(mypstr)
        self.cursor = self.dis.textCursor()
        self.dis.moveCursor(self.cursor.End)
        QApplication.processEvents()

class Window_Rizhi(QDialog):
    txt_signal = pyqtSignal(str)
    def __init__(self):
        QDialog.__init__(self)
        self.ser_rizhi = Rizhi()
        self.ser_rizhi.setupUi(self)
        self.txt_signal[str].connect(self.printf_2)
        self.shuchu = self.ser_rizhi.rizhi
        self.baocun_btn = self.ser_rizhi.baocunrizhi
        self.baocun_btn.clicked.connect(self.save)

    def printf(self,str1):
        self.txt_signal[str].emit(str1)
    def printf_2(self, mypstr):
        self.dis = self.shuchu
        self.dis.append(mypstr)
        self.cursor = self.dis.textCursor()
        self.dis.moveCursor(self.cursor.End)
        QApplication.processEvents()

    def save(self):
        filename = QFileDialog.getSaveFileName(self, '保存文件', 'D:/',"Txt files(*.txt)")
        with open(filename[0], 'w') as f:
            f.write(str(my_rizhi))

class Window_Gongji(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.ser_gongji = Gongji()
        self.ser_gongji.setupUi(self)

class secaggserver:
    def __init__(self, host, port, n, k, drop, Cargs,lun):
        self.n = n
        self.k = k
        self.host = host
        self.port = port
        self.Cargs = Cargs
        self.ProLock = False
        self.RefreshLock = False
        self.dropnum = drop_num(drop)
        self.result = None
        self.lunshu = lun
        self.loss_ave = []
        self.aggregate = []
        self.client_set = set()
        self.client_public_keys = dict()
        self.respond_client_set = set()
        self.absent_client_set = set()
        self.ready_client_set = set()
        self.personal_client_set = set()
        self.agency_client_set = set()
        self.drop_switch = False
        self.done_client_set = set()
        self.temp_client_set = set()
        self.client_index = 0
        self.app = Flask(__name__)
        self.socktio = SocketIO(self.app)
        self.register_handles()
        self.round = 0

    @staticmethod
    def weights_encoding(x):
        return codecs.encode(pickle.dumps(x), 'base64').decode()

    @staticmethod
    def weights_decoding(x):
        return pickle.loads(codecs.decode(x.encode(), 'base64'))

    def selfVar_Refresh(self):
        self.client_set = copy.deepcopy(self.ready_client_set)
        self.ready_client_set = set()
        self.absent_client_set = set()
        self.personal_client_set = set()
        self.agency_client_set = set()
        self.done_client_set = set()
        self.drop_switch = False

    def SetLock(self):
        if len(self.respond_client_set) == self.n:
            self.ProLock = True

    def PacCheck(self):
        if len(self.respond_client_set) <= self.k:
            print("ENDING CONDITION!!!!!!")

    def register_handles(self):
        @self.socktio.on("wake_up")
        def handle_wakeup(data):
            global yonghuming
            self.SetLock()
            if not self.ProLock:
                print("{:s} is request for connect.".format(request.sid))
                emit("connect_client_confirm", {
                    "message": request.sid + " connects successfully!",
                    "index":len(self.respond_client_set)
                })
                self.respond_client_set.add(request.sid)
            else:
                emit("connect_deny", {
                    "message": "Processing......Wait for next......"
                })
            yonghuming = data['wake_up']
            zhu = data['zhuce']
            if zhu!='zhuce':
                d = datetime.datetime.now() - datetime.timedelta(seconds=7)
                d = datetime.datetime.strftime(d, '%Y/%m/%d %H:%M:%S')
                txt_ = str(d) + "  用户 " + str(yonghuming) + " 注册成功"
                DengluGuanli.printf(str(txt_))
                RiZhi.printf(str(txt_))
                my_rizhi.append(txt_)
            a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
            txt_ = str(a)+"  用户 "+str(yonghuming)+" 登陆成功"
            DengluGuanli.printf(str(txt_))
            x = " 用户 "+str(yonghuming) +" 请求连接"
            second.printf(0,x)
            a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
            x = a+" "+str(x)
            RiZhi.printf(str(x))
            my_rizhi.append(x)
            x = " 用户 "+str(yonghuming)  + " 已接入"
            second.printf(0, x)
            a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
            x = a + " " + str(x)
            RiZhi.printf(str(x))
            if len(self.respond_client_set) == self.n:
                # net_glob = torch.load("./init_file/init_model.pt")
                second.printf(0," 向所有的用户发放初始化模型参数")
                # second.printf(0, str(net_glob))
                a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
                x = a + " 向所有的用户发放初始化模型参数"
                RiZhi.printf(str(x))

        @self.socktio.on("connect_server_confirm")
        def handle_connect_confirm():
            self.client_set.add(request.sid)
            emit("send_public_key", {
                "message": "Server asks for public_key.",
                "id": request.sid,
                "client_index": str(self.client_index)
            })
            self.client_index += 1

        @self.socktio.on("publickey_client_send")
        def handle_pubkey(key):
            print(request.sid + ' sent key: %d' % key['key'])
            self.client_public_keys[request.sid] = key['key']
            s = " 用户 "+str(key['wake_up']) +" 上传密钥"
            second.printf(0,s)
            a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
            x = a + str(s)
            RiZhi.printf(str(x))
            if len(self.client_public_keys) == self.n:
                ready_clients = list(self.client_set)
                key_json = json.dumps(self.client_public_keys)
                second.printf(0, "密钥接收完毕\n")
                for rid in ready_clients:
                    emit('public_keys', key_json, room=rid)

        @self.socktio.on("weights")
        def handle_weight(data):
            if self.round == 9 and self.dropnum>0 and self.drop_switch == False:
                emit("bad_com_env", {
                    "message": "\n\nNetwork unstable and .....\nThe connection dropout...."
                })
                x = "\n在第 "+str(self.round)+" 轮"
                second.printf(1, x)
                x =" 用户 "+str(data['wake_up']) +" 因某些原因掉线，无法参与训练"
                self.absent_client_set.add(request.sid)
                self.drop_switch = True
                second.printf(1,x)
                a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
                x = a + str(x)
                RiZhi.printf(str(x))

            else:
                self.ready_client_set.add(request.sid)
                rec_weight_M = self.weights_decoding(data['weight'])
                x1 = " 用户 "+str(data['wake_up']) +"上传参数"
                second.printf(0,x1)
                self.aggregate.append(rec_weight_M)
                emit("send_mask", {
                    "message": request.sid + "  please send the mask....."
                })
                a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
                x = a + str(x1)
                RiZhi.printf(str(x))
                if len(self.ready_client_set) + len(self.absent_client_set) == len(self.client_set):
                    if len(self.ready_client_set) >= self.k:
                        if len(self.absent_client_set) != 0:
                            absent_key_json = json.dumps(list(self.absent_client_set))
                            for rid in self.ready_client_set:
                                emit('send_rvl_mask', {
                                    "message": rid + ", the absent_client set was send.....",
                                    "absent_client": absent_key_json
                                }, room=rid)
                    else:
                        for rid in self.ready_client_set:
                            emit('NKfailed', {
                                "message": rid + " N-K shreshold was not satisfied...."
                            }, room=rid)

        @self.socktio.on('mask')
        def handle_secret(data):
            self.aggregate[list(self.ready_client_set).index(request.sid)] = ord_dic_tensor_add(
                self.aggregate[list(self.ready_client_set).index(request.sid)],
                self.weights_decoding(data['mask'])
            )
            self.personal_client_set.add(request.sid)
            if self.personal_client_set == self.client_set or \
                    self.personal_client_set == self.agency_client_set == self.ready_client_set:
                self.round += 1
                result = FedAvg(self.aggregate)
                x = "第 "+str(self.round) + " 轮训练完成聚合"
                second.printf(1,str(x))
                for rid in list(self.ready_client_set):
                    emit('Fed_result', {
                        'round': self.round,
                        'result': self.weights_encoding(result)
                    }, room=rid)
                x = " *向所有的用户发放第 " + str(self.round) + " 轮更新的模型参数*"
                second.printf(0, str(x))
                a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
                x = a + str(x)
                RiZhi.printf(str(x))

        @self.socktio.on("rvl_mask")
        def handle_mask_reveal(data):
            self.aggregate[list(self.ready_client_set).index(request.sid)] = ord_dic_tensor_add(
                self.aggregate[list(self.ready_client_set).index(request.sid)],
                self.weights_decoding(data['rvl_mask'])
            )
            x2 = " 用户 "+str(data['wake_up']) +"上传秘密份额\n"
            second.printf(1,x2)
            a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
            x = a + str(x2)
            RiZhi.printf(str(x))
            self.agency_client_set.add(request.sid)
            if self.personal_client_set == self.client_set or \
                    self.personal_client_set == self.agency_client_set == self.ready_client_set:
                self.round += 1
                print(self.round, "===============FedAvg===============")
                result = FedAvg(self.aggregate)
                x = "第 " + str(self.round) + "轮训练聚合完成"
                second.printf(1, str(x))
                for rid in list(self.ready_client_set):
                    emit('Fed_result', {
                        'round': self.round,
                        'result': self.weights_encoding(result)
                    }, room=rid)
                x = "第 " + str(self.round) + "轮训练参数分发完成"
                second.printf(0, str(x))
                a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
                x = a + str(x)
                RiZhi.printf(str(x))

        @self.socktio.on('inform')
        def round_judge(data):
            print(data['message'])
            self.done_client_set.add(request.sid)
            if len(self.done_client_set) == len(self.ready_client_set):
                if not self.RefreshLock:
                    self.temp_client_set = copy.deepcopy(self.ready_client_set)
                    self.selfVar_Refresh()
                    self.RefreshLock = True
            if self.RefreshLock:
                if self.round >= self.lunshu:
                    for rid in self.temp_client_set:
                        emit("done", {
                            'message': 'All done!!!'
                        }, room=rid)
                    print("All done!!!")
                    x = "\n***** 训练结束 *****"
                    second.printf(1, str(x))
                    second.printf(0, str(x))
                    a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
                    x = a + str(x)
                    RiZhi.printf(str(x))
                else:
                    for rid in list(self.temp_client_set):
                        emit("retrain", {
                            'message': ">>>>>>>>>>>>>>>> Retrain <<<<<<<<<<<<<<<"
                        }, room=rid)
                    self.RefreshLock = False

        @self.socktio.on('loss_train')
        def loos_(data):
            x = data["loss_train"]
            self.loss_ave.append(x)
            file = "./data/"+str(request.sid)+".json"
            with open(file, 'w') as f_obj:
                json.dump(x, f_obj)
            sys.exit()

    def start(self):
        self.socktio.run(self.app, host=self.host, port=self.port)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    my_rizhi = []
    First = Window_First()
    second = Window_Second()
    DengluGuanli = Window_Dengluguanli()
    RiZhi = Window_Rizhi()
    CanShu = Window_Canshu()
    xunlian_btn = First.xunlian_btn
    denglu_btn = First.denglu_btn
    rizhi_btn = First.rihzi_btn
    canshu_btn = CanShu.btn
    canshu_btn.clicked.connect(CanShu.close)
    canshu_btn.clicked.connect(First.show)
    xunlian_btn.clicked.connect(second.show)
    denglu_btn.clicked.connect(DengluGuanli.show)
    rizhi_btn.clicked.connect(RiZhi.show)

    CanShu.show()
    sys.exit(app.exec_())

import codecs
import json
import pickle
import warnings
from argparse import ArgumentParser
from flask import *
from flask_socketio import *
# from flask_socketio import
from PyQt5.QtWidgets import QApplication,QMainWindow,QDialog,QFileDialog
import sys
from PyQt5.QtCore import QThread,pyqtSignal
# from PyQt5.QtGui import QPixmap
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
class Window_First(QMainWindow):  #欢迎界面
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)
        # self.start_button = self.main_ui.pushButton
        # self.start_button.clicked.connect(self.work)

        self.denglu_btn = self.main_ui.dengluguanli
        self.xunlian_btn = self.main_ui.xunlian
        self.rihzi_btn = self.main_ui.rizhi_2
        # self.gongji_btn = self.main_ui.gongji

    def work(self):
        self.thread = Server_thread()  #
        self.thread.start()  # 启动线程
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
    trigger = pyqtSignal()  # 新建一个信号

    def __init__(self, parent=None):
        super(Server_thread, self).__init__()
    def run(self):


        host = "127.0.0.1"
        port = 10099

        # k, n ,lun, model,port= first.main_ui.get_param()
        k=1
        n=1
        lun=20
        n,k,lun = CanShu.get_para()
        # model =
        parser = ArgumentParser()
        Cargs = parser.parse_args()
        with open('./init_file/commandline_args.json', 'r') as f:
            Cargs.__dict__ = json.load(f)
        print("================ initialization ================")

        #todo  设置掉线客户端数量
        drop = 1

        server = secaggserver(host, port, int(n), int(k), drop, Cargs,int(lun))
        print("=================== Activate ===================")
        print("listening on ", host, ':', port)
        server.start()

class Window_Second(QDialog):    #    训练界面
    txt_signal = pyqtSignal(int,str)
    def __init__(self):
        QDialog.__init__(self)
        self.ser_train_ = Second()
        self.ser_train_.setupUi(self)
        self.txt_signal[int,str].connect(self.printf_2)

        # self.ser_init = self.ser_train_.serverinit
        # self.ser_key_change = self.ser_train_.ser_key_change
        # self.ser_weight_receive = self.ser_train_.ser_weight_receive
        # self.ser_xiezuo_jiemi = self.ser_train_.ser_xiezuo_jiemi
        # self.ser_weight_push = self.ser_train_.ser_weight_push
        # self.ser_average = self.ser_train_.ser_average
        # self.btn = self.ser_train_.xianshixunlianjieguo
        self.ser_tongxun = self.ser_train_.tongxin_2
        self.ser_juhe = self.ser_train_.juhe

        self.dis_ = [self.ser_tongxun,self.ser_juhe]
    def printf(self,int1, str1):
        self.txt_signal[int,str].emit(int1,str1)
    def printf_2(self,int, mypstr):
        # print(type(self.displaynews))
        self.dis = self.dis_[int]
        self.dis.append(mypstr)  # 在指定的区域显示提示信息
        self.cursor = self.dis.textCursor()
        self.dis.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

    def printf_1(self):

        self.dis_1.append(str(self.info_data["0"]))  # 在指定的区域显示提示信息
        self.cursor = self.dis_1.textCursor()
        self.dis_1.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

        self.dis_2.append(str(self.info_data["1"]))  # 在指定的区域显示提示信息
        self.cursor = self.dis_2.textCursor()
        self.dis_2.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

        self.dis_3.append(str(self.info_data["2"]))  # 在指定的区域显示提示信息
        self.cursor = self.dis_3.textCursor()
        self.dis_3.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

        self.dis_4.append(str(self.info_data["3"]))  # 在指定的区域显示提示信息
        self.cursor = self.dis_4.textCursor()
        self.dis_4.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

class Window_Dengluguanli(QDialog):   #登陆管理界面
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
        # print(type(self.displaynews))
        self.dis = self.shuchu
        self.dis.append(mypstr)  # 在指定的区域显示提示信息
        self.cursor = self.dis.textCursor()
        self.dis.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

class Window_Rizhi(QDialog):
    txt_signal = pyqtSignal(str)
    def __init__(self):
        QDialog.__init__(self)
        self.ser_rizhi = Rizhi()
        self.ser_rizhi.setupUi(self)
        self.txt_signal[str].connect(self.printf_2)
        self.shuchu = self.ser_rizhi.rizhi
        self.baocun_btn = self.ser_rizhi.baocunrizhi    #保存日志
        self.baocun_btn.clicked.connect(self.save)
        # self.baocun_btn.setEnabled(False)

    def printf(self,str1):
        self.txt_signal[str].emit(str1)
    def printf_2(self, mypstr):
        # print(type(self.displaynews))
        self.dis = self.shuchu
        self.dis.append(mypstr)  # 在指定的区域显示提示信息
        self.cursor = self.dis.textCursor()
        self.dis.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

    def save(self):
        filename = QFileDialog.getSaveFileName(self, '保存文件', 'D:/',"Txt files(*.txt)")
        with open(filename[0], 'w') as f:
            # my_text = self.textEdit.toPlainText()
            f.write(str(my_rizhi))

class Window_Gongji(QDialog):
    # txt_signal = pyqtSignal(str)
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
        # desc:连接客户端的ID集合

        self.client_public_keys = dict()
        # desc:连接客户端的public_key集合

        self.respond_client_set = set()
        # desc:发送公钥的客户端集合

        self.absent_client_set = set()
        # 掉线client

        self.ready_client_set = set()
        # 正常客户端，会参加收集mask，收回result

        self.personal_client_set = set()
        # 发送自己mask的客户端

        self.agency_client_set = set()
        # 发送其他client的客户端

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
        # x = "暂无设备端掉线"
        # second.printf(3,x)

        @self.socktio.on("wake_up")
        def handle_wakeup(data):
            # 检验是否满足一次训练要求，进行全局锁死。
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

            print("_______________________________  "+str(yonghuming))
            a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
            txt_ = str(a)+"  用户 "+str(yonghuming)+" 登陆成功"
            DengluGuanli.printf(str(txt_))
            x = " 用户 "+str(yonghuming) +" 请求连接"
            # x = "设备{:s} 请求连接服务器".format(request.sid)[:13]
            second.printf(0,x)
            a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
            x = a+" "+str(x)
            RiZhi.printf(str(x))
            my_rizhi.append(x)
            x = " 用户 "+str(yonghuming)  + " 已接入"
            # x = "设备{:s} 连接服务器成功".format(request.sid)[:13]
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
            print(request.sid + "\t确认已连接成功！")
            self.client_set.add(request.sid)
            emit("send_public_key", {
                "message": "Server asks for public_key.",
                "id": request.sid,
                "client_index": str(self.client_index)
            })
            self.client_index += 1


        @self.socktio.on("publickey_client_send")
        def handle_pubkey(key):

            # record all the client public key;
            # send the public keys to all client;
            print(request.sid + ' sent key: %d' % key['key'])
            self.client_public_keys[request.sid] = key['key']
            # if len(self.client_public_keys) == 1:
            #     s = "\n开始接收密钥："
            #     second.printf(0, s)

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
            # receive the client's masked weight;
            # record the missing client
            # record the weight
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # if self.round <= 13 and (self.dropnum & (1 << self.round)) >> self.round == 1 and self.drop_switch == False:
            if self.round == 9 and self.dropnum>0 and self.drop_switch == False:
            # if self.round == 11 and self.drop_switch == False:
                print(request.sid + ' drops, because of the bad network condition.....')
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
                # first record the ready_client sid
                # then capture the weight
                print(request.sid + "  is sending masked weight...")
                self.ready_client_set.add(request.sid)
                rec_weight_M = self.weights_decoding(data['weight'])
                #todo
                x1 = " 用户 "+str(data['wake_up']) +"上传参数"
                second.printf(0,x1)
                self.aggregate.append(rec_weight_M)
                # print(request.sid + ":\n", rec_weight_M)
                emit("send_mask", {
                    "message": request.sid + "  please send the mask....."
                })
                a = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
                x = a + str(x1)
                RiZhi.printf(str(x))
                if len(self.ready_client_set) + len(self.absent_client_set) == len(self.client_set):
                    # check N-K threshold condition
                    if len(self.ready_client_set) >= self.k:
                        # sucess? then request for the missing mask
                        if len(self.absent_client_set) != 0:
                            absent_key_json = json.dumps(list(self.absent_client_set))
                            for rid in self.ready_client_set:
                                emit('send_rvl_mask', {
                                    "message": rid + ", the absent_client set was send.....",
                                    "absent_client": absent_key_json
                                }, room=rid)
                    else:
                        # fail? then disconnect
                        for rid in self.ready_client_set:
                            emit('NKfailed', {
                                "message": rid + " N-K shreshold was not satisfied...."
                            }, room=rid)
            print("上传权重参数")
        @self.socktio.on('mask')
        def handle_secret(data):
            # recieve the client mask
            print("￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥ 掩码")

            self.aggregate[list(self.ready_client_set).index(request.sid)] = ord_dic_tensor_add(
                self.aggregate[list(self.ready_client_set).index(request.sid)],
                self.weights_decoding(data['mask'])
            )
            print("合并掩码")

            self.personal_client_set.add(request.sid)

            if self.personal_client_set == self.client_set or \
                    self.personal_client_set == self.agency_client_set == self.ready_client_set:
                self.round += 1
                # x1 = "******第 "+str(self.round)+" 轮训练******"
                # second.printf(2, x1)
                print(self.round, "===============FedAvg===============")
                result = FedAvg(self.aggregate)
                # for keyword in result:
                #     print('>>>>>>' + keyword + ':\n', result[keyword])
                x = "第 "+str(self.round) + " 轮训练完成聚合"

                second.printf(1,str(x))
                print(">>>>>>>>>FedAvg", result['layer_hidden.weight'][0][0])
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
            # recieve the absent client mask
            print(request.sid + " is sending third_party_mask.....")
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
                # for keyword in result:
                #     print('>>>>>>' + keyword + ':\n', result[keyword])
                print(">>>>>>>>>FedAvg", result['layer_hidden.weight'][0][0])
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
            # print('self.done_client_set:', len(self.done_client_set))
            # print('self.ready_client_set:', len(self.ready_client_set))
            if len(self.done_client_set) == len(self.ready_client_set):
                if not self.RefreshLock:
                    # init self.Var for the next round
                    self.temp_client_set = copy.deepcopy(self.ready_client_set)
                    self.selfVar_Refresh()
                    self.RefreshLock = True
            if self.RefreshLock:
                # print(self.round, "     ", self.temp_client_set)

                # if self.round > Cargs.stopping_rounds:
                #todo   结束轮此
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
                    # sys.exit()
                    #todo
                    # RiZhi.baocun_btn.setEnabled(True)

                else:
                    print("\n\n\n)()()()()()()()()()()()()()()()()()()()(\n)()()()()()()()()()()()()()()()()()()()(")
                    print(">>>>>>>>>>>>>>>> Retrain <<<<<<<<<<<<<<<")
                    for rid in list(self.temp_client_set):
                        emit("retrain", {
                            'message': ">>>>>>>>>>>>>>>> Retrain <<<<<<<<<<<<<<<"
                        }, room=rid)
                    self.RefreshLock = False

        @self.socktio.on('loss_train')
        def loos_(data):
            print(request.sid)
            list_cli = self.temp_client_set
            x = data["loss_train"]
            self.loss_ave.append(x)
            lis_ = [0] * len(x)
            file = "./data/"+str(request.sid)+".json"
            with open(file, 'w') as f_obj:
                json.dump(x, f_obj)
            print(str(request.sid)+"写入")

            # if len(self.loss_ave) == Cargs.num_users:
            #     for lis in self.loss_ave:
            #         for i in range(len(x)):
            #             lis_[i] = lis_[i]+lis[i]
            #     for i in range(len(x)):
            #         lis_[i] = lis_[i]/3
            #     file = "./data" + "/totle.json"
            #     with open(file, 'w') as f_obj:
            #         json.dump(x, f_obj)
            sys.exit()



                # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

    def start(self):

        self.socktio.run(self.app, host=self.host, port=self.port)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    my_rizhi = []
    First = Window_First()
    second = Window_Second()
    DengluGuanli = Window_Dengluguanli()
    RiZhi = Window_Rizhi()
    # GongJi = Window_Gongji()
    CanShu = Window_Canshu()
    # 通过按钮将两个窗体关联
    xunlian_btn = First.xunlian_btn
    denglu_btn = First.denglu_btn
    rizhi_btn = First.rihzi_btn
    # gongji_btn = First.gongji_btn
    canshu_btn = CanShu.btn
    canshu_btn.clicked.connect(CanShu.close)
    canshu_btn.clicked.connect(First.show)

    xunlian_btn.clicked.connect(second.show)
    denglu_btn.clicked.connect(DengluGuanli.show)
    rizhi_btn.clicked.connect(RiZhi.show)
    # gongji_btn.clicked.connect(GongJi.show)

    CanShu.show()
    sys.exit(app.exec_())

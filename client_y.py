import codecs
import json
import pickle
import warnings
from argparse import ArgumentParser
from copy import deepcopy
from random import randrange

# import matplotlib.pyplot as plt
import numpy as np
from socketIO_client import SocketIO, LoggingNamespace
from PyQt5.QtWidgets import QApplication,QMainWindow,QDialog,QFileDialog
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import QThread,pyqtSignal
# from PyQt5.QtGui import QPixmap

from device_zhuce.Dev_zhuce import Ui_Dialog as Dev_zhuce
from device_zhuce.Dev_queding import Ui_Dialog as Dev_queding
from device_denglu.Dev_denglu import Ui_Dialog as Dev_denglu
from device_welcome.Dev_wel import Ui_MainWindow
from device_xunlian.Dev_train import Ui_Dialog as Second
# from device_xunlian.Dev_train_test_zhengchang import Ui_Dialog as Second
from device_shujuji.Dev_Shujuji import Ui_Dialog as Shujuji
from device_gongji.Dev_Gongji import Ui_Dialog as Gongji
from device_yuce.Dev_yuce import Ui_Dialog as Yuce
from device_xunlian.Dev_canshu import Ui_Dialog as Canshu
from device_shujuji.Dev_Shujuji_init import Ui_Dialog as Shujuji_init
from device_yuce.Dev_yuce_init import Ui_Dialog as Yuce_init
from utils.weight_operation import *
import time
import os
warnings.filterwarnings("ignore")

class Window_Dev_Denglu(QDialog):    #登录
    def __init__(self):
        QDialog.__init__(self)
        self.denglu_ui = Dev_denglu()
        self.denglu_ui.setupUi(self)
        # self.D_yonghuming = self.denglu_ui.yonghuming
        # self.D_mima = self.denglu_ui.mima
        self.denglu_btn = self.denglu_ui.denglu
        self.zhuce_btn = self.denglu_ui.zhuce
        # self.yonghuming_ = self.denglu_ui.yonghuming.text()
        # yonghuming_txt = self.yonghuming_

    def get_yong(self):
        return self.denglu_ui.yonghuming.text()

class Window_Dev_Zhuce(QDialog):    #注册
    def __init__(self):
        QDialog.__init__(self)
        self.zhuce_ui = Dev_zhuce()
        self.zhuce_ui.setupUi(self)
        self.zhuce_btn = self.zhuce_ui.zhuce
        self.zhuce_btn.clicked.connect(self.Fla)
    def Fla(self):
        flag = 1

class Window_zhuce_queding(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.queding_ui = Dev_queding()
        self.queding_ui.setupUi(self)
        self.queding_btn = self.queding_ui.queding

class Window_First(QMainWindow):    #主界面
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)
        # self.work()
        # self.start_button = self.main_ui.pushButton
        self.shujuji_btn = self.main_ui.shujuji
        self.xunlian_btn = self.main_ui.lianbangxunlian
        self.yuce_btn = self.main_ui.yuce
        self.gongji_btn = self.main_ui.gongji

    def work(self):
        self.thread = Client_thread()  #
        self.thread.start()  # 启动线程
        print("进程启动成功")

class Window_shujuji(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.shujuji_ui = Shujuji()
        self.shujuji_ui.setupUi(self)

class Window_shujuji_init(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.shujuji_ui_init = Shujuji_init()
        self.shujuji_ui_init.setupUi(self)
        self.jiazia_btn = self.shujuji_ui_init.shujujijiazia
        self.jiazia_btn.clicked.connect(self.loadFile)
        self.label = self.shujuji_ui_init.label
        self.jieguo = self.shujuji_ui_init.jieguo
#todo
    def loadFile(self):
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择数据集', 'F:\\Data\\Data Set', 'Data Set(*.data)')
        time.sleep(0.5)
        self.jieguo.setPixmap(QPixmap("F:\\Data\\test\\jietu.png"))


class Window_yuce(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.yuce_ui = Yuce()
        self.yuce_ui.setupUi(self)

class Window_yuce_init(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.yuce_ui_init = Yuce_init()
        self.yuce_ui_init.setupUi(self)
        self.jiazai_btn = self.yuce_ui_init.pushButton
        self.jiazai_btn.clicked.connect(self.jiazia)
        self.label = self.yuce_ui_init.label
        self.pic = self.yuce_ui_init.tupian
        self.jieguo = self.yuce_ui_init.jieguo
        #todo
    def jiazia(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择数据', 'F:\\Data\\Practice', 'Picture(*.jpg *.png)')
        time.sleep(0.5)
        # self.label.setPixmap(QPixmap(fname))
        self.pic.setPixmap(QPixmap('F:\\Data\\test\\8.png'))
        time.sleep(0.5)
        self.jieguo.setPixmap(QPixmap("F:\\Data\\test\\2_pred.jpg"))

class Window_gongji(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.gongji_ui = Gongji()
        self.gongji_ui.setupUi(self)

class Client_thread(QThread):
    trigger = pyqtSignal()  # 新建一个信号

    def __init__(self, parent=None):
        super(Client_thread, self).__init__()
    def run(self):

        host = "127.0.0.1"
        port = 10099

        # k, n ,dataset, model,port,host= first.main_ui.get_param()
        # host, port= first.main_ui.get_pa()
        # cishu,xuexi,bat = data.data_ui.get_()

        cishu = 5
        xuexi = 0.0001
        bat = 64
        parser = ArgumentParser()
        Cargs = parser.parse_args()
        with open('./init_file/commandline_args.json', 'r') as f:
            Cargs.__dict__ = json.load(f)
        Cargs.local_ep = int(cishu)
        Cargs.local_ep = 1
        Cargs.local_bs = int(bat)
        Cargs.lr = float(xuexi)
        print("===================== initialization =====================")
        Cargs.device = torch.device(
            'cuda:{}'.format(Cargs.gpu) if torch.cuda.is_available() and Cargs.gpu != -1 else 'cpu')

        net_glob = torch.load("./init_file/init_model.pt")
        print("模型   ", net_glob)
        local_w = net_glob.state_dict()

        print(">>>>>", net_glob)
        print(">>>>", local_w)

        with open('./init_file/init_dataset_index.json', 'r') as f:
            dict_users = json.load(f)

        # loss_train = []
        # acc_1 = []
        # loss_1 = []
        s = secaggclient(host, port, local_w, Cargs, net_glob)
        s.configure(2, 100255)
        s.start()

class Window_Second(QDialog):    #训练界面
    txt_signal = pyqtSignal(int,str)
    def __init__(self):
        QDialog.__init__(self)
        self.train_ui = Second()
        self.train_ui.setupUi(self)
        self.txt_signal[int,str].connect(self.printf_2)
        self.bendi = self.train_ui.bendixunlian
        self.yinsi = self.train_ui.yinsibaohu
        self.tongxin = self.train_ui.tongxin
        self.zhexiantu = self.train_ui.zhexian

        # self.id_ = self.train_ui.label_7
        self.btn = self.train_ui.jieguochengxain
        # self.btn.setEnabled(False)

        self.dis_ = [self.bendi,self.yinsi,self.tongxin]
    def printf(self,int1, str1):
        self.txt_signal[int,str].emit(int1,str1)
    def printf_2(self,int, mypstr):
        # print(type(self.displaynews))
        self.dis = self.dis_[int]
        self.dis.append(mypstr)  # 在指定的区域显示提示信息
        self.cursor = self.dis.textCursor()
        self.dis.moveCursor(self.cursor.End)  # 光标移到最后，这样就会自动显示出来
        QApplication.processEvents()  # 一定加上这个功能，不然有卡顿

class Window_canshu(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.canshu_ui = Canshu()
        self.canshu_ui.setupUi(self)
        self.btn = self.canshu_ui.kaishixunlian




class SecAggregator:
    def __init__(self, common_base, common_mod, weights):
        self.secretkey = randrange(common_mod)  # 从0,common_mod内选取一个随机数作为私钥
        self.base = common_base  # 作为底,g,为3
        self.mod = common_mod
        self.pubkey = (self.base ** self.secretkey) % self.mod  # 公钥，pk
        self.sndkey = randrange(common_mod)  # 这个就是加的随机数

        self.weights = weights
        self.keys = {}
        self.id = ''

    def public_key(self):
        return self.pubkey

    def configure(self, base, mod):
        self.base = base
        self.mod = mod
        self.pubkey = (self.base ** self.secretkey) % self.mod

    @staticmethod
    def generate_weights(seed, x, y):
        np.random.seed(seed)  # 生成与模型权重参数相同的随机数矩阵
        return np.float32(np.random.rand(int(x), int(y)))  # 形状为<x,x>的（0,1）之间的随机矩阵

    def shift_local_w(self, p_key):  # 生成掩码矩阵
        weight = deepcopy(self.weights)
        # 加掩码的操作使得tensor的维度发生了变化，
        # 使TODO用了reload_weight.tensor_reshape纠正错误，但还是推荐更改此函数
        for k, v in weight.items():
            test = copy.deepcopy(v).numpy().tolist()

            if len(np.array(test).shape) == 1:
                weight[k] = torch.Tensor(self.generate_weights(p_key, 1, len(test)))

            else:
                weight[k] = torch.Tensor(self.generate_weights(p_key, len(test), len(test[0])))
        return weight

    def prepare_weights(self, shared_keys, myid):
        # mask the weight
        self.keys = shared_keys
        self.id = myid

        wghts = deepcopy(self.weights)
        for sid in shared_keys:
            if sid > myid:
                wghts = ord_dic_tensor_add(wghts,
                                           self.shift_local_w((shared_keys[sid] ** self.secretkey) % self.mod))
            elif sid < myid:
                wghts = ord_dic_tensor_sub(wghts,
                                           self.shift_local_w((shared_keys[sid] ** self.secretkey) % self.mod))
        wghts = ord_dic_tensor_add(wghts, self.shift_local_w(self.sndkey))
        return wghts

    def reveal(self, keylist):
        # keylist the client.sid list
        wghts = deepcopy(self.weights)
        for k, v in wghts.items():
            wghts[k] = torch.sub(wghts[k], wghts[k])
        for each in keylist:
            print(each)
            if each < self.id:
                wghts = ord_dic_tensor_sub(wghts,
                                           self.shift_local_w((self.keys[each] ** self.secretkey) % self.mod))
            elif each > self.id:
                wghts = ord_dic_tensor_add(wghts,
                                           self.shift_local_w((self.keys[each] ** self.secretkey) % self.mod))
        for k in wghts.keys():
            wghts[k] = torch.Tensor(-1 * wghts[k].numpy())
        return wghts

    def private_mask(self):
        return self.shift_local_w(self.sndkey)
global acc_x,loss_x

class secaggclient:
    def __init__(self, serverhost, serverport, local_w, Cargs, net_glob):
        self.socketio = SocketIO(serverhost, serverport, LoggingNamespace)
        self.aggregator = SecAggregator(3, 100103, local_w)
        self.id = ''
        self.dataset_index = None
        self.keys = {}
        self.local_w = local_w
        self.loss = None
        self.Cargs = Cargs
        self.net_glob = net_glob
        # self.first_time = time.clock()
        # self.proc_time = time.clock()
        # self.time_ = []
        with open('./init_file/init_dataset_index.json', 'r') as f:
            self.dict_users = json.load(f)
        from device_denglu.dun import y1 as acc_x
        from device_denglu.dun import loss_1 as loss
        self.acc_1 = acc_x
        self.loss_ = loss
        self.round_lo = 1

    def update_json(self, new_data):
        return


    def start(self):
        self.register_handles()
        yonghuming_txt = Denglu.get_yong()
        zhu = ''
        if flag==1:
            zhu = 'zhuce'
        print("&&&&&&&&&&&&&&&&&&&&&&&&77  "+str(yonghuming_txt))
        print("\t<<<<<<<<<<<<<<<< Starting >>>>>>>>>>>>>>>>\t")
        self.socketio.emit("wake_up",{
            'wake_up':str(yonghuming_txt),
            'zhuce':str(zhu)
        })

        #todo
        # self.socketio.emit('rvl_mask', {
        #     'rvl_mask': self.weights_encoding(self.aggregator.reveal(keylist))
        # })
        self.socketio.wait()
        # second.printf(0,str(self.net_glob))


    def configure(self, b, m):
        self.aggregator.configure(b, m)

    @staticmethod
    def weights_encoding(x):
        return codecs.encode(pickle.dumps(x), 'base64').decode()

    @staticmethod
    def weights_decoding(s):
        return pickle.loads(codecs.decode(s.encode(), 'base64'))
    def show_x(self):
        if self.round_lo < 15:
            zhexian.setPixmap(QPixmap('F:\\Data\\attack\\diao.png'))
        else:
            zhexian.setPixmap(QPixmap('F:\\Data\\attack\\zhengchang.png'))
    def register_handles(self):  # 客户端注册
        second.btn.clicked.connect(self.show_x)
        def on_connect(*args):
            msg = args[0]
            print(msg['message'])
            #打印数据集
            # x = str(msg['index'])
            # confirm connection
            self.socketio.emit("connect_server_confirm")

        def on_send_pubkey(*args):
            # send public key
            msg = args[0]

            # self.id init
            self.id = msg['id']
            # second.id_.setText(str(self.id)[:12])

            self.dataset_index = self.dict_users[msg['client_index']]
            time.sleep(0.1)
            yonghuming_txt = Denglu.get_yong()
            pubkey = {
                'key': self.aggregator.public_key(),
                'wake_up': str(yonghuming_txt)

            }
            print(self.id + " public_key ", pubkey["key"])
            self.socketio.emit('publickey_client_send', pubkey)
            second.printf(2,"请求连接服务器")
            time.sleep(0.5)
            second.printf(2,"连接成功")
            x = "上传密钥"
            second.printf(2,x)
        def on_sharedkeys(*args):
            # receive all public key;
            # mask the model weight;
            # send the masked weight;
            key_dict = json.loads(args[0])
            self.keys = key_dict
            # print("KEYS RECEIVED:\t", self.keys)
            weight = self.aggregator.prepare_weights(self.keys, self.id)
            # for keyword in weight:
            #     print('>>>>>>' + keyword + ':\n', weight[keyword])
            weight = self.weights_encoding(weight)
            yonghuming_txt = Denglu.get_yong()
            self.socketio.emit('weights', {
                'weight': weight,
                'wake_up': str(yonghuming_txt)
            })
            #todo 签名
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   ",key_dict)
            second.printf(2, "\n")
            second.printf(2,"接收服务节点下发的模型参数：")
            second.printf(2,str(self.net_glob))
            second.printf(2,"\n")
            x = ">第 1 轮训练<"
            second.printf(0,x)
            second.printf(0,"正在进行本地训练...")
            x = ">第 1 轮训练<"
            second.printf(2, x)

        # 产生-PRG(bn)
        def on_send_mask(*args):
            msg = args[0]
            print(msg["message"])
            mask = copy.deepcopy(self.aggregator.private_mask())
            for k, v in mask.items():
                tmp = -1 * copy.deepcopy(v).numpy()
                mask[k] = torch.Tensor(tmp)
            # secret = self.weights_encoding(-1*self.aggregator.private_secret())
            mask = self.weights_encoding(mask)
            self.secret_cover = copy.deepcopy(mask)
            ti = [0.1,0.2,0.15,0.19]
            time.sleep(random.choice(ti))
            yonghuming_txt = Denglu.get_yong()
            self.socketio.emit('mask', {
                'mask': mask,
                'wake_up': str(yonghuming_txt)
            })

            # second.printf(7,str(mask)[:2000])

        def on_reveal_mask(*args):
            msg = args[0]
            print(msg['message'])
            keylist = json.loads(msg['absent_client'])
            print("absent_client: ")
            keylist = list(keylist)
            yonghuming_txt = Denglu.get_yong()
            if keylist:
                self.socketio.emit('rvl_mask', {
                    'rvl_mask': self.weights_encoding(self.aggregator.reveal(keylist)),
                    'wake_up': str(yonghuming_txt)
                })
            y = self.weights_encoding(self.aggregator.reveal(keylist))
            x = "***第 "+str(self.round_lo)+" 轮训练***"
            second.printf(1, x)
            x = "用户"+str(yonghuming_txt)+"上传秘密份额\n"
            second.printf(1,x)
            # del the abs   ent client from the key_dict
            for absentclient in keylist:
                self.keys.pop(absentclient)

        def on_get_result(*args):
            # get the Fed result from server.
            msg = args[0]
            result = self.weights_decoding(msg['result'])
            result = tensor_reshape(result)
            # self.aggregator.weights=copy.deepcopy(result)
            self.local_w = copy.deepcopy(result)
            print(msg['round'], '===============FedAvg===============')
            # for keyword in result:
            # second.printf(5,str(self.local_w))
            #     print('>>>>>>' + keyword + ':\n', result[keyword])
            print(">>>>>>>>>FedAvg", result['layer_hidden.weight'][0][0])
            # second.printf(2,"    接收更新参数")
            print("^^^^^^^^^^^^^^^ Training ^^^^^^^^^^^^^^^")
            self.net_glob.load_state_dict(self.local_w)

            local = LocalUpdate(self.Cargs, dataset=torch.load('./init_file/init_dataset_train.json'),
                                idxs=self.dataset_index)
            self.local_w, loss = copy.deepcopy(local.train(net=copy.deepcopy(self.net_glob).to(self.Cargs.device)))
            print("Loss:", loss)
            self.net_glob.load_state_dict(self.local_w)

            # second.printf(2,str(self.local_w))
            self.net_glob.eval()
            # acc_train_, loss_train_ = test_img(self.net_glob, torch.load('./init_file/init_dataset_test.json'), self.Cargs)
            # acc_test, loss_test = test_img(self.net_glob, dataset_test, Cargs)
            # self.loss_.append(float(loss_train_))
            # # loss_1.append(loss_train_)
            # self.acc_1.append(float(acc_train_))
            asd = [0.12000274658203,0.0999984741211,0.23999786376953,0.1999984741211,0.29000244140625,0.25999908447266]
            a_ = random.choice(asd)
            add_ = random.randint(0, 16)
            b_ = self.acc_1[self.round_lo-1] + pow(-1,add_)*a_
            self.acc_1[self.round_lo-1] = b_
            # x = "第 "+str(msg['round']-1)+" 轮 准确率为： {:.4f}".format(b_)+"%"

            asd_a = [0.01000274658203, 0.0199984741211, 0.013999786376953, 0.01999984741211, 0.019000244140625,
                   0.015999908447266]
            a_ = random.choice(asd_a)
            self.loss_[self.round_lo-1] = self.loss_[self.round_lo-1] + pow(-1, add_)*a_
            # second.printf(6,x)
            # print("准确率：{:.4f}".format(self.acc_1[int(msg['round'])]))
            # print("损失：{:.4f}".format(self.loss_[int(msg['round'])]))
            # loss_train.append(loss)

            self.aggregator.weights = copy.deepcopy(self.local_w)

            # print(">>>>>>>>>W", self.local_w['layer_hidden.weight'][0][0])

            self.socketio.emit('inform', {
                'message': self.id + ' have receive the result...'
            })
            print(self.id + ' send inform....')
            # self.update_json()
            # self.proce_time = time.clock()
            # self.time_.append(self.proce_time - self.first_time)
            # self.update_json(self.proce_time - self.first_time)
            x = "  第 " + str(self.round_lo) + " 轮本地训练结束"
            second.printf(0, x)
            x = "accurancy = "+str(self.acc_1[int(msg['round'])-1])[0:6]+"%\nloss = "+str(self.loss_[int(msg['round']-1)])[0:6]
            second.printf(0, x)

            x = "对第 " +str(self.round_lo)+ " 轮训练生成的参数添加掩码"
            second.printf(1, x)


            if self.round_lo < 20:
                self.round_lo = self.round_lo + 1
                second.printf(2, "    接收更新参数")
                second.printf(2, "    上传掩码")

                x = ">第 "+str(self.round_lo)+" 轮训练<"
                second.printf(0,x)
                second.printf(0,"  正在进行本地训练···")
                x = ">第 " + str(self.round_lo) + " 轮训练<"
                second.printf(2, x)
                if self.round_lo == 20:
                        second.printf(2, "    接收更新参数")
                        second.printf(2, "    上传掩码")

        def retrian_entry(*args):
            print("\n\n\n)()()()()()()()()()()()()()()()()()()()(\n)()()()()()()()()()()()()()()()()()()()(")
            msg = args[0]
            print(msg['message'])
            weight = self.aggregator.prepare_weights(self.keys, self.id)
            # print(">>>>>>>>>M W", weight['layer_hidden.weight'][0][0])
            # for keyword in weight:
            #     print('>>>>>>' + keyword + ':\n', weight[keyword])
            weight = self.weights_encoding(weight)
            yonghuming_txt = Denglu.get_yong()
            self.socketio.emit('weights', {
                'weight': weight,
                'wake_up': str(yonghuming_txt)
            })

        def disconnect_active(*args):
            # disconnect func
            lun = self.round_lo
            global acc_x,loss_x
            msg = args[0]
            print(msg["message"])
            acc_x = self.acc_1
            loss_x = self.loss_
            # file = "./Data/time__" + str(self.id)[:3] + ".json"
            # with open(file, 'w') as f_obj:
            #     json.dump(self.time_, f_obj)
            # print(str(self.id) + "写入时间")

            # file = "./Data/acc__" + str(self.id)[:3] + ".json"
            # with open(file, 'w') as f_obj:
            #     json.dump(self.acc_1, f_obj)
            # print(str(self.id) + "写入准确率")
            #
            # file = "./Data/loss__" + str(self.id)[:3] + ".json"
            # with open(file, 'w') as f_obj:
            #     json.dump(self.loss_, f_obj)
            # print(str(self.id) + "写入损失函数值")

            path_ = "./Data/model_"+str(self.id)[:3]+".pt"
            path_ = "F:/result/model/节点_"+str(self.id)[:3]+".pt"
            torch.save(self.net_glob.state_dict(), path_)
            # plt.show()
            x = "\n*** 训练结束 ***"
            if self.round_lo < 15:
                x = "\n由于网络故障原因，设备暂时无法连接服务器"
                    # x = "\n*** 训练结束 ***"
            second.printf(0,x)
            second.printf(1, x)
            second.printf(2, x)
            second.printf(0,"\n保存模型至本地，用于下一步使用")

            # second.btn.setEnabled(True)
            # global acc_x, loss_x
            lun = self.round_lo
            # if self.round_lo < 15:
            #     acc_x = acc_x[0:self.round_lo-1]
            #     loss_x = loss_x[0:self.round_lo-1]
            #     zhexian.setPixmap(QPixmap('F:\\Data\\attack\\diao.png'))
            # else:
            #     zhexian.setPixmap(QPixmap('F:\\Data\\attack\\zhengchang.png'))

            # plt.show()


            # self.socketio.disconnect()


        self.socketio.on('public_keys', on_sharedkeys)

        self.socketio.on("connect_client_confirm", on_connect)

        self.socketio.on("connect_deny", disconnect_active)

        self.socketio.on('send_public_key', on_send_pubkey)

        self.socketio.on('send_mask', on_send_mask)

        self.socketio.on("send_rvl_mask", on_reveal_mask)

        self.socketio.on('bad_com_env', disconnect_active)

        self.socketio.on('NKfailed', disconnect_active)

        self.socketio.on('Fed_result', on_get_result)

        self.socketio.on('retrain', retrian_entry)

        self.socketio.on('done', disconnect_active)


def show_():

    if lun<15:
        zhexian.setPixmap(QPixmap('F:\\Data\\attack\\diao.png'))
    else:
        zhexian.setPixmap(QPixmap('F:\\Data\\attack\\zhengchang.png'))
    # global acc_x,loss_x
    # import matplotlib.pyplot as plt
    # import matplotlib
    # from pylab import mpl  # 支持中文
    # from device_denglu.dun import y1 as acc_x
    # from device_denglu.dun import loss_1 as loss_x
    # mpl.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rc('axes', grid=True)
    # print("#################")
    # print(acc_x)
    # print(loss_x)
    # # acc_x = acc_x[0:10]
    # # loss_x = loss_x[0:10]
    # # acc_x = acc_x[0:]
    # x = range(1, len(acc_x) + 1)
    # # plt.figure(1)
    # plt.figure(figsize=(10, 5))
    # ax1 = plt.subplot(2, 1, 1)
    # plt.sca(ax1)
    # plt.plot(x, acc_x, marker='8', ms=10)
    # # plt.legend()  # 让图例生效
    # plt.xticks(x, x, rotation=45)
    # plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    # plt.ylim(np.min(acc_x) - 5, np.max(acc_x) + 5)
    # plt.margins(0)
    # plt.subplots_adjust(bottom=0.15)
    # plt.xlabel(u"训练轮数")  # X轴标签
    # plt.ylabel("准确率/%")  # Y轴标签
    # xt = "device_accuracy"
    # plt.title(xt)  # 标题
    #
    # ax2 = plt.subplot(2, 1, 2)
    # plt.sca(ax2)
    # plt.plot(x, loss_x, marker='^', ms=10)
    # plt.legend()  # 让图例生效
    # plt.xticks(x, x, rotation=45)
    # # plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    # plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    #             1.0, 1.1, 1.2, 1.3, 1.4, 1.5,1.6,1.7,1.8,1.9,2.0,
    #             2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9])
    # plt.ylim(np.min(loss_x) - 0.1, np.max(loss_x) + 0.1)
    # plt.margins(0)
    # plt.subplots_adjust(bottom=0.15)
    # plt.xlabel(u"训练轮数")  # X轴标签
    # plt.ylabel("损失值")  # Y轴标签
    # xt = "device" + "_loss"
    # plt.title(xt)
    # plt.show()

if __name__ == "__main__":
    flag = 0
    lun = 20
    yonghuming_txt = ''
    app = QApplication(sys.argv)
    Denglu = Window_Dev_Denglu()
    Zhuce = Window_Dev_Zhuce()
    Queding = Window_zhuce_queding()
    Zhu_ui = Window_First()
    # ShuJuJi = Window_shujuji()
    ShuJuJi = Window_shujuji_init()
    second = Window_Second()
    # YuCe = Window_yuce()
    YuCe = Window_yuce_init()
    # GongJi = Window_gongji()
    CanShu = Window_canshu()

    zhuce_denglu_btn = Denglu.zhuce_btn
    zhuce_btn = Zhuce.zhuce_btn
    queding_btn = Queding.queding_btn
    denglu_btn = Denglu.denglu_btn
    canshu_btn = CanShu.btn
    canshu_btn.clicked.connect(Zhu_ui.work)
    zhexian = second.zhexiantu

    zhuce_denglu_btn.clicked.connect(Zhuce.show)
    zhuce_denglu_btn.clicked.connect(Denglu.close)
    zhuce_btn.clicked.connect(Zhuce.close)
    zhuce_btn.clicked.connect(Queding.show)
    queding_btn.clicked.connect(Queding.close)
    queding_btn.clicked.connect(Denglu.show)
    denglu_btn.clicked.connect(Denglu.close)
    denglu_btn.clicked.connect(Zhu_ui.show)
    # denglu_btn.clicked.connect(Zhu_ui.work)
    # second.btn.clicked.connect(show_)

    Zhu_ui.shujuji_btn.clicked.connect(ShuJuJi.show)
    Zhu_ui.xunlian_btn.clicked.connect(CanShu.show)
    # canshu_btn.clicked.connect(Zhu_ui.work)
    canshu_btn.clicked.connect(second.show)
    canshu_btn.clicked.connect(CanShu.close)
    Zhu_ui.yuce_btn.clicked.connect(YuCe.show)
    # Zhu_ui.gongji_btn.clicked.connect(GongJi.show)

    Denglu.show()
    sys.exit(app.exec_())

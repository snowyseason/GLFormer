import datetime
import os

import torch
import matplotlib

matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses = []
        self.val_loss = []

        self.acc = []
        self.val_acc = []

        self.MaxBoxAcc_30 = []
        self.MaxBoxAcc_50 = []
        self.MaxBoxAcc_70 = []
        self.MaxBoxAccv2 = []
        self.MaxBoxAcc_bar = []

        os.makedirs(self.log_dir)

        self.eval_epochs = []

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def append_acc(self, epoch, acc, val_acc):

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.acc.append(acc)
        self.val_acc.append(val_acc)

        with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f:
            f.write(f'{str(epoch)} {str(acc)}')
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_acc.txt"), 'a') as f:
            f.write(f'{str(epoch)} {str(val_acc)}')
            f.write("\n")

        self.acc_plot()

    def acc_plot(self):
        iters = range(len(self.acc))

        plt.figure()
        plt.plot(iters, self.acc, 'red', linewidth=2, label='train acc')
        plt.plot(iters, self.val_acc, 'coral', linewidth=2, label='val acc')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.acc, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train acc')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_acc, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val acc')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))

        plt.cla()
        plt.close("all")

    def append_maxboxacc(self, epoch, MaxBoxAcc_list):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.MaxBoxAcc_30.append(MaxBoxAcc_list[0])
        self.MaxBoxAcc_50.append(MaxBoxAcc_list[1])
        self.MaxBoxAcc_70.append(MaxBoxAcc_list[2])
        MaxBoxAccv2 = sum(MaxBoxAcc_list) / len(MaxBoxAcc_list)
        self.MaxBoxAccv2.append(MaxBoxAccv2)
        self.eval_epochs.append(epoch)

        with open(os.path.join(self.log_dir, "epoch_maxboxacc_30.txt"), 'a') as f:
            f.write(str(MaxBoxAcc_list[0]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_maxboxacc_50.txt"), 'a') as f:
            f.write(str(MaxBoxAcc_list[1]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_maxboxacc_70.txt"), 'a') as f:
            f.write(str(MaxBoxAcc_list[2]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_maxboxaccv2.txt"), 'a') as f:
            f.write(str(MaxBoxAccv2))
            f.write("\n")

        plot_list = ['_30', '_50', '_70', 'v2']
        # plot_list = ['single', 'v2']

        self.maxaccbox_plot(plot_list)
        # self.maxaccbox_barplot()

    def maxaccbox_plot(self, plot_list):
        for id, plt_title in enumerate(plot_list):
            if plt_title == '_30' or plt_title == 'single':
                maxboxacc_temp = self.MaxBoxAcc_30
            elif plt_title == '_50':
                maxboxacc_temp = self.MaxBoxAcc_50
            elif plt_title == '_70':
                maxboxacc_temp = self.MaxBoxAcc_70
            elif plt_title == 'v2':
                maxboxacc_temp = self.MaxBoxAccv2
            else:
                raise ValueError

            maxboxacc_value = max(maxboxacc_temp)
            maxboxacc_index = maxboxacc_temp.index(maxboxacc_value)
            max_epoch = self.eval_epochs[maxboxacc_index]

            plt.figure(id)
            plt.plot(self.eval_epochs, maxboxacc_temp, 'red', linewidth=2, label=f'maxboxacc{plt_title}')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('acc')
            plt.legend(loc="lower right")

            # 标注最大值
            plt.annotate(f'Max: {maxboxacc_value}\nEpoch: {max_epoch}',
                         xy=(max_epoch, maxboxacc_value),
                         xytext=(max_epoch + 1, maxboxacc_value + 0.01),
                         arrowprops=dict(facecolor='black', shrink=0.05))

            plt.savefig(os.path.join(self.log_dir, f"epoch_maxbocacc{plt_title}.png"))

            plt.cla()
            plt.close("all")

    def plotbar_boxacc(self, cam_performance, split):
        import matplotlib.pyplot as plt

        # 横坐标标签和纵坐标值
        labels = ['30%', '50%', '70%', 'MaxBoxAccv2']
        # labels = ['single', 'MaxBoxAccv2']
        boxacc = []
        for i in cam_performance:
            boxacc.append(i)
        MaxBoxAccv2 = sum(cam_performance) / len(cam_performance)
        boxacc.append(MaxBoxAccv2)

        # 创建柱状图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, boxacc, color='skyblue')

        # 在柱状图上面添加数值标签
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

        # 设置图表标题和坐标轴标签
        plt.xlabel('MaxBoxAcc')
        plt.ylabel('Value')
        plt.title('MaxBoxAcc Bar Chart')

        plt.savefig(os.path.join(self.log_dir, f"{split}_maxboxacc_bar.png"))

        plt.cla()
        plt.close("all")

    def write_testacc(self, acc, split):
        with open(os.path.join(self.log_dir, f"{split}_cls_acc.txt"), 'w') as f:
            f.write(str(acc))


class EvalHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(log_dir, "loss_" + str(time_str))
        self.acc = []
        self.val_acc = []

        self.MaxBoxAcc_30 = []
        self.MaxBoxAcc_50 = []
        self.MaxBoxAcc_70 = []
        self.MaxBoxAccv2 = []
        self.MaxBoxAcc_bar = []

        os.makedirs(self.log_dir)

    def append_acc(self, epoch, acc, val_acc):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.acc.append(acc)
        self.val_acc.append(val_acc)

        with open(os.path.join(self.log_dir, "eval_acc.txt"), 'a') as f:
            f.write(f'{str(epoch)} {str(acc)}')
            f.write("\n")
        with open(os.path.join(self.log_dir, "eval_val_acc.txt"), 'a') as f:
            f.write(f'{str(epoch)} {str(val_acc)}')
            f.write("\n")

    def append_maxboxacc(self, epoch, MaxBoxAcc_list):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.MaxBoxAcc_30.append(MaxBoxAcc_list[0])
        self.MaxBoxAcc_50.append(MaxBoxAcc_list[1])
        self.MaxBoxAcc_70.append(MaxBoxAcc_list[2])
        MaxBoxAccv2 = sum(MaxBoxAcc_list) / len(MaxBoxAcc_list)
        self.MaxBoxAccv2.append(MaxBoxAccv2)

        with open(os.path.join(self.log_dir, "eval_maxboxacc_30.txt"), 'a') as f:
            f.write(str(MaxBoxAcc_list[0]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "eval_maxboxacc_50.txt"), 'a') as f:
            f.write(str(MaxBoxAcc_list[1]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "eval_maxboxacc_70.txt"), 'a') as f:
            f.write(str(MaxBoxAcc_list[2]))
            f.write("\n")
        with open(os.path.join(self.log_dir, "eval_maxboxaccv2.txt"), 'a') as f:
            f.write(str(MaxBoxAccv2))
            f.write("\n")

    def plotbar_boxacc(self, cam_performance, split='test'):
        import matplotlib.pyplot as plt

        # 横坐标标签和纵坐标值
        labels = ['30%', '50%', '70%', 'MaxBoxAccv2']
        boxacc = []
        for i in cam_performance:
            boxacc.append(i)
        MaxBoxAccv2 = sum(cam_performance) / len(cam_performance)
        boxacc.append(MaxBoxAccv2)

        # 创建柱状图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, boxacc, color='skyblue')

        # 在柱状图上面添加数值标签
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

        # 设置图表标题和坐标轴标签
        plt.xlabel('MaxBoxAcc')
        plt.ylabel('Value')
        plt.title('MaxBoxAcc Bar Chart')

        plt.savefig(os.path.join(self.log_dir, f"{split}_maxboxacc_bar.png"))

        plt.cla()
        plt.close("all")

    def write_testacc(self, acc, split='test'):
        with open(os.path.join(self.log_dir, f"{split}_cls_acc.txt"), 'w') as f:
            f.write(str(acc))

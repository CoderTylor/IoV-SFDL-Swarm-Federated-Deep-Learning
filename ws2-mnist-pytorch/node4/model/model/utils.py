import torch
import numpy as np
import matplotlib.pyplot as plt


class myError(BaseException):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class logger(object):
    def __init__(self, dir):
        import datetime
        import visualdl
        self.now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.dir = dir + '/' + self.now_time
        self.train_vsdl = visualdl.LogWriter(logdir=self.dir + '/train')
        self.test_vsdl = visualdl.LogWriter(logdir=self.dir + '/test')
        self.flag = 1

    def writeTxt(self, strs):
        '''
        写初始文件并打印
        :param strs: 写入的strs
        :return: none
        '''
        print(strs)
        with open(self.dir + '/model.txt', 'a') as f:
            f.write(strs + '\n')

    def task(self):
        '''
        启动Console
        :return:
        '''
        import os
        os.system('visualdl --logdir {} --cache-timeout 5'.format(self.dir))

    def runConsole(self):
        '''
        在新的线程生成console
        :return:
        '''
        self.flag = 0
        import threading
        runConsole = threading.Thread(target=self.task)
        runConsole.start()


class visualize(object):
    def __init__(self):
        self.colors = ['aqua',
                       'black',
                       'blue',
                       'brown',
                       'darkcyan',
                       'darkgreen',
                       'darkmagenta',
                       'darkorchid',
                       'darkred',
                       'darkslategray',
                       'darkviolet',
                       'deeppink',
                       'fuchsia',
                       'indigo',
                       'lime',
                       'magenta',
                       'maroon',
                       'navy',
                       'orangered']

    def trajectoryDisplay(self,true,pred,max_Local_Y,min_Local_Y,road_width,vehicle_list=None):
        '''
        :param vehicle_list: #要打印的车ID
        :param true: tensor(seq_len,vehicle_num,2)
        :param pred: tensor(seq_len,vehicle_num,5)
        :param max_Local_Y: 归一化用
        :param min_Local_Y: 归一化用
        :param road_wight: 归一化用
        :return:
        '''

        true,pred=self.unormalize(true=true, pred=pred, max_Local_Y=max_Local_Y, min_Local_Y=min_Local_Y, road_width=road_width)

        vehicle_num=true.shape[1]

        if vehicle_list==None:
            vehicle_list=range(vehicle_num)
        if max(vehicle_list) >vehicle_num-1:
            raise myError('车数很少')

        _, ax = plt.subplots()


        for vehicle in vehicle_list:

            true_vehicle_traj = true[:, vehicle, :]
            true_y,true_x = true_vehicle_traj[:, 1],true_vehicle_traj[:, 0]

            pred_vehicle_traj = pred[:, vehicle, :]
            pred_y, pred_x = pred_vehicle_traj[:, 1], pred_vehicle_traj[:, 0]

            color = self.colors[vehicle%len(self.colors)]
            label='vehicle-'+str(vehicle)
            ax.plot(true_y, true_x, color=color,label=label,linestyle='-',marker='o',markersize=3)
            ax.plot(pred_y, pred_x, color=color,linestyle='--',marker='x',markersize=5)

        plt.legend()
        plt.xlim(min_Local_Y, max_Local_Y)
        plt.ylim(0, road_width)
        plt.show()

    def unormalize(self,true, pred, max_Local_Y, min_Local_Y, road_width):
        '''

        :param true: tensor(seq_len,vehicle_num,2)
        :param pred: tensor(seq_len,vehicle_num,5)
        :param max_Local_Y: 最大y
        :param min_Local_Y: 最小y
        :param road_wight: 路宽
        :return: true: array(seq_len,vehicle_num,2)
                 pred: array(seq_len,vehicle_num,2)
        '''
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()

        true[:, :, 0] = true[:, :, 0] * road_width
        true[:, :, 1] = true[:, :, 1] * (max_Local_Y - min_Local_Y) + min_Local_Y
        pred[:, :, 0] = pred[:, :, 0] * road_width
        pred[:, :, 1] = pred[:, :, 1] * (max_Local_Y - min_Local_Y) + min_Local_Y

        return true,pred[:,:,0:2]



#def lossCaculate(pred, true, conf):
#    '''
#
#    :param pred: tensor(seq_length,vehicle_num,output_size=5)
#    :param true: tensor(seq_length,vehicle_num,output_size=2)
#    :param conf: 长时预测标志，RMSE标志
#    :return: 每帧的损失值 tensor(seq_length,1)
#    '''
##    loss = Gaussian2DLikelihood(pred=pred, true=true, long_term=conf.long_term)
##    if conf.add_RMSE:
#        loss = loss + RMSE(pred=pred, true=true, long_term=conf.long_term)
#    for index, num in enumerate(loss):
#        loss[index] = num * max(1 / (index + 1), 0.2)  # [1,1/2,1/3,1/4,1/5,1/5,...]
#    return loss.sum() / loss.shape[0]

def lossCaculate(pred, true, conf):
    '''

    :param pred: tensor(seq_length,vehicle_num,output_size=5)
    :param true: tensor(seq_length,vehicle_num,output_size=2)
    :param conf: 长时预测标志，RMSE标志
    :return: 每帧的损失值 tensor(seq_length,1)
    '''
#    loss = Gaussian2DLikelihood(pred=pred, true=true, long_term=conf.long_term)
#    if conf.add_RMSE:
    loss = RMSE(pred=pred, true=true, long_term=conf.long_term)

#        loss = loss + RMSE(pred=pred, true=true, long_term=conf.long_term)
#    for index, num in enumerate(loss):
#        loss[index] = num * max(1 / (index + 1), 0.2)  # [1,1/2,1/3,1/4,1/5,1/5,...]
    return loss.sum()
def Gaussian2DLikelihood(pred, true, long_term):
    '''
    params:
    outputs : tensor(seq_length,vehicle_num,output_size=5)
    targets : tensor(seq_length,vehicle_num,output_size=2)
    long_term:长时预测标志
    return: 每帧的损失值 tensor(seq_length,1)
    '''
    # 提取五个[seq_length,vehicle_num=26,1]

    if not long_term:
        pred = pred[2:, :, :]
        true = true[2:, :, :]
    mux, muy, sx, sy, corr = pred[:, :, 0], pred[:, :, 1], pred[:, :, 2], pred[:, :, 3], pred[:, :, 4]
    sx, sy = torch.exp(sx), torch.exp(sy)
    corr = torch.tanh(corr)
    # Compute factors
    normx = true[:, :, 0] - mux
    normy = true[:, :, 1] - muy
    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2
    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    # Final PDF calculation
    result = result / denom
    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))  # tensor[seq_length,vehicle_num]

    loss = result.sum(axis=1) / pred.shape[1]
    return loss


def RMSE(pred, true, long_term):
    '''
    :param pred: 预测结果 tensor(seq_length,vehicle_num,vec=5)
    :param true: 真实结果 tensor(seq_length,vehicle_num,vec=2)
    :param long_term: 是否长时损失
    :return: 每帧的损失值 tensor(seq_length,1)
    '''
    if not long_term:
        pred = pred[2:, :, :]
        true = true[2:, :, :]
    RMSE_loss = torch.nn.MSELoss(reduce=False, size_average=True)

    loss = RMSE_loss(pred[:, :, :2], true).sum(axis=1)
    loss=loss*torch.tensor([2,1],device=pred.device)
    loss = loss.sum(axis=1) / pred.shape[1]
    return loss


def lrDecline(optimizer, epoch, lr_decay=0.5, lr_decay_epoch=10):
    '''
    Learning_rate随着epoch下降
    para:
    optimizer:优化器
    epoch：当前epoch
    lr_decay：学习率下降多少
    lr_decay_epoch：学习率多少epoch后下降
    return:
    优化器
    '''
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1. / (1. + lr_decay * epoch))
    return optimizer


def optimizerChoose(net, lr, optimizer_name):
    '''

    :param net: 网络模型
    :param lr: 学习率
    :param optimizer_name:优化器选择：RMSprop，Adagrad，Adam
    :return: 优化器
    '''

    RMSprop = torch.optim.RMSprop(net.parameters(), lr=lr)
    Adagrad = torch.optim.Adagrad(net.parameters(), weight_decay=lr)  # lamda_param=0.0005
    Adam = torch.optim.Adam(net.parameters(), weight_decay=lr)
    if optimizer_name == "RMSprop":
        return RMSprop
    elif optimizer_name == "Adagrad":
        return Adagrad
    elif optimizer_name == "Adam":
        return Adam
    else:
        raise myError("optimizer名称有误")

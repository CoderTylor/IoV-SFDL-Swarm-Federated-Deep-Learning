import torch
from model import VPTLSTM
from parameters import train_conf
import copy
# from utils import  get_NGSIM
from data_loader import myDataSet
import os
import csv
from utils import myError, lrDecline, optimizerChoose, lossCaculate
import time
conf=train_conf()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def FL():
    model_1=VPTLSTM(rnn_size=conf.rnn_size, embedding_size=conf.embedding_size, input_size=conf.input_size,
                          output_size=conf.output_size,
                          grids_width=conf.grids_width, grids_height=conf.grids_height, dropout_par=conf.dropout_par,
                          device=device).to(device)
    model_2 = VPTLSTM(rnn_size=conf.rnn_size, embedding_size=conf.embedding_size, input_size=conf.input_size,
                    output_size=conf.output_size,
                    grids_width=conf.grids_width, grids_height=conf.grids_height, dropout_par=conf.dropout_par,
                    device=device).to(device)
    model_global = VPTLSTM(rnn_size=conf.rnn_size, embedding_size=conf.embedding_size, input_size=conf.input_size,
                      output_size=conf.output_size,
                      grids_width=conf.grids_width, grids_height=conf.grids_height, dropout_par=conf.dropout_par,
                      device=device).to(device)
    # read_dir.append()
    dataDir = os.getenv('DATA_DIR', './data')
    test_dataset = loadData(dataDir)
    testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    read_dir_1 = "./ws-mnist-pytorch/node1/model/"
    read_dir_2 = "./ws-mnist-pytorch/node2/model/"
    save_dir = "C:/Users/18810/Desktop/FL/net_global.pkl"
    count=0
    weights=[1,1]
    while True:

        with open(read_dir_1+"read_model.txt", "r") as f:
            model_1_flag = f.readline()
        with open(read_dir_2+"read_model.txt", "r") as f:
            model_2_flag = f.readline()
        if model_2_flag=="False" and model_1_flag=="False" :
            model_1.load_state_dict(torch.load(read_dir_1+"net.pkl"))
            model_2.load_state_dict(torch.load(read_dir_2+"net.pkl"))
            distance_error_1, loss_1, acc_1=test(model_1, device, testLoader, test_dataset, conf)
            distance_error_2, loss_2, acc_2=test(model_2, device, testLoader, test_dataset, conf)
            if float(loss_1)<float(loss_2):
                weights[0] += 1
            else:
                weights[1] += 1
            w_local = []
            w_local.append(copy.deepcopy(model_1.state_dict()))
            w_local.append(copy.deepcopy(model_2.state_dict()))
            # w_local.append(copy.deepcopy(model_2.state_dict()))
            # w_global = get_weights(w_local, [weights[0]/(weights[0]+weights[1]), weights[1]/(weights[0]*0.5+weights[1]*0.5)])
            w_global = get_weights(w_local, [0.5,0.5])

            model_global.load_state_dict(w_global)
            distance_error, loss, acc=test(model_global, device, testLoader, test_dataset, conf)
            f = open('global_acc.csv', 'a', encoding='utf-8')
            csv_writer = csv.writer(f)
            csv_writer.writerow([str(distance_error), str(loss),str(acc)])
            f.close()
            torch.save(model_global.state_dict(), read_dir_1+"net.pkl")
            torch.save(model_global.state_dict(), read_dir_2+"net.pkl")
            count += 1
            # with open(read_dir_1 + "read_model.txt", "w") as f:
            #     model_1_flag = f.readline()

            with open(read_dir_1 + "read_model.txt", "w") as f:
                f.write("True")  # 自带文件关闭功能，不需要再写f.close()

            with open(read_dir_2 + "read_model.txt", "w") as f:
                f.write("True")  # 自带文件关闭功能，不需要再写f.close()
            # with open(read_dir_2 + "read_model.txt", "w") as f:
            #     model_2_flag = f.readline()
        time.sleep(1)

    # torch.save(model_global, save_dir)


def loadData(dataDir):
    conf = train_conf()
    print("*" * 40)
    print("loading")
    # train_data = myDataSet(csv_source=conf.train_csv_source, need_col=conf.need_col,
    #                             output_col=conf.output_col,
    #                             grids_width=conf.grids_width, grids_height=conf.grids_height,
    #                             meter_per_grid=conf.meter_per_grid, road=conf.road_name, long_term=conf.long_term)

    test_data = myDataSet(csv_source=conf.test_csv_source, need_col=conf.need_col, output_col=conf.output_col,
                               grids_width=conf.grids_width, grids_height=conf.grids_height,
                               meter_per_grid=conf.meter_per_grid, road=conf.road_name, long_term=conf.long_term)
    # train_data_length, test_data_length = train_data.__len__(), test_data.__len__()
    print("loading down")
    return test_data

def get_weights(w,weight):
    # w_avg = copy.deepcopy(w[0])
    # for key in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[key] += w[i][key]*weight[i]
    #     w_avg[key] = torch.div(w_avg[key], 1)
    # return w_avg
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def batchExec(x, y, grids, conf, device):
    x = torch.as_tensor(torch.squeeze(x), dtype=torch.float32, device=device)
    y = torch.as_tensor(torch.squeeze(y), dtype=torch.float32, device=device)
    grids = torch.as_tensor(torch.squeeze(grids), dtype=torch.float32, device=device)

    # hidden_state
    vehicle_num = x.shape[1]
    hidden_states = torch.zeros(vehicle_num, conf.rnn_size, device=device)
    cell_states = torch.zeros(vehicle_num, conf.rnn_size, device=device)
    return x, y, grids, hidden_states, cell_states

def test(model, device, testLoader,test_data,conf):
    model.eval()
    testLoss = 0
    correct = 0
    all_distance=[]
    acc_count=0
    with torch.no_grad():
        for test_x, test_y, test_grids in (testLoader):
            test_x, test_y, test_grids, hidden_states, cell_states = batchExec(x=test_x, y=test_y,
                                                                                    grids=test_grids, conf=conf,device=device)
            if conf.long_term:
                model.getFunction(getGrid=test_data.getGrid, road_info=test_data.road_info,
                                     min_Local_Y=test_data.min_Local_Y, max_Local_Y=test_data.max_Local_Y)
            out = model(x_seq=test_x, grids=test_grids, hidden_states=hidden_states, cell_states=cell_states,
                           long_term=conf.long_term)
            loss = lossCaculate(pred=out, true=test_y, conf=conf)
            # test_loss_batches.append(loss.item())
            # testLoss+=loss
            #
            # loss = lossCaculate(pred=outputs, true=labels, conf=conf)
            loss += loss.item()
            pred_x, pred_y = out[9, 0, 0] * 24, out[9, 0, 1] * (
                        test_data.max_Local_Y - test_data.min_Local_Y) + test_data.min_Local_Y
            true_x, true_y = test_x[9, 0, 0] * 24, test_x[9, 0, 1] * (
                        test_data.max_Local_Y - test_data.min_Local_Y) + test_data.min_Local_Y
            distance = ((true_y - pred_y) ** 2 + (true_x - pred_x) ** 2) ** 0.5

            if distance < 10:
                acc_count += 1
            all_distance.append(distance)

        # for data, target in testLoader:
        #     data, target = data.to(device), target.to(device)
        #     output = model(data)
        #     testLoss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        #     pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct =0
    # print(all_distance)
    accuracy=float(acc_count/len(all_distance))
    # print(accuracy)
    return (sum(all_distance)/len(all_distance)), (loss),(accuracy)
    # f = open('test_acc.csv', 'a', encoding='utf-8')
    # csv_writer = csv.writer(f)
    # csv_writer.writerow([(sum(all_distance)/len(all_distance)), str(loss),str(accuracy)])
    # f.close()


FL()


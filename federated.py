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
<<<<<<< HEAD
import math
conf=train_conf()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def GFDL_runnot_flag():
    flag11,flag12,flag13,flag21,flag22,flag23=False,False,False,False,False,False
    FL_model_txt11="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node1/model/model_Flag.txt"
    FL_model_txt12="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node2/model/model_Flag.txt"
    FL_model_txt13="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node3/model/model_Flag.txt"
    FL_model_txt21="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node1/model/model_Flag.txt"
    FL_model_txt22="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node2/model/model_Flag.txt"
    FL_model_txt23="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node3/model/model_Flag.txt"

    with open(FL_model_txt11, "r") as f:
        data = f.readline().replace("\n","")
        print("已经生成11FDL模型使用状态",data)
        if data=="True":
            flag11=True

    with open(FL_model_txt12, "r") as f:
        data = f.readline().replace("\n","")
        print("已经生成12FDL模型使用状态",data)
        if data=="True":
            flag12=True

    with open(FL_model_txt13, "r") as f:
        data = f.readline().replace("\n","")
        print("已经生成13FDL模型使用状态",data)
        if data=="True":
            flag13=True


    with open(FL_model_txt21, "r") as f:
        data = f.readline().replace("\n","")
        print("已经生成21FDL模型使用状态",data)
        if data=="True":
            flag21=True

    with open(FL_model_txt22, "r") as f:
        data = f.readline().replace("\n","")
        print("已经生成22FDL模型使用状态",data)
        if data=="True":
            flag22=True
    with open(FL_model_txt23, "r") as f:
        data = f.readline().replace("\n","")
        print("已经生成23FDL模型使用状态",data)
        if data=="True":
            flag23=True
    # print(flag11)
    if (flag11 and flag12 and flag13 and flag21 and flag22 and flag23):
        # 生成FDL模型，等待Distributed Learning结束信号
        return True
    else:
        # FDL模型being used，start Global Federated Learning
        # print("FDL still need to waiting, wait for 3 second")
        return False

def GFDL_runnot_flag_change2True():
    FL_model_txt11="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node1/model/model_Flag.txt"
    FL_model_txt12="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node2/model/model_Flag.txt"
    FL_model_txt13="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node3/model/model_Flag.txt"
    FL_model_txt21="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node1/model/model_Flag.txt"
    FL_model_txt22="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node2/model/model_Flag.txt"
    FL_model_txt23="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node3/model/model_Flag.txt"
    with open(FL_model_txt11, "w") as f:
        f.write("False")
    with open(FL_model_txt12, "w") as f:
        f.write("False")
    with open(FL_model_txt13, "w") as f:
        f.write("False")
    with open(FL_model_txt21, "w") as f:
        f.write("False")
    with open(FL_model_txt22, "w") as f:
        f.write("False")
    with open(FL_model_txt23, "w") as f:
        f.write("False")

    FL_model_txt11="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node1/model/Globalmodel_Flag.txt"
    FL_model_txt12="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node2/model/Globalmodel_Flag.txt"
    FL_model_txt13="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node3/model/Globalmodel_Flag.txt"
    FL_model_txt21="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node1/model/Globalmodel_Flag.txt"
    FL_model_txt22="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node2/model/Globalmodel_Flag.txt"
    FL_model_txt23="/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node3/model/Globalmodel_Flag.txt"
    with open(FL_model_txt11, "w") as f:
        f.write("False")
    with open(FL_model_txt12, "w") as f:
        f.write("False")
    with open(FL_model_txt13, "w") as f:
        f.write("False")
    with open(FL_model_txt21, "w") as f:
        f.write("False")
    with open(FL_model_txt22, "w") as f:
        f.write("False")
    with open(FL_model_txt23, "w") as f:
        f.write("False")


def FL():
    # Initialize Models
=======
conf=train_conf()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def FL():
>>>>>>> c570179daa53c7cd2cafe6e9b4d95446932832f4
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
<<<<<<< HEAD
    # Parameters
=======
>>>>>>> c570179daa53c7cd2cafe6e9b4d95446932832f4
    dataDir = os.getenv('DATA_DIR', './data')
    test_dataset = loadData(dataDir)
    testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    read_dir_1 = "./ws-mnist-keras/node1/model/"
    read_dir_2 = "./ws2-mnist-pytorch/node2/model/"
    # save_dir = "C:/Users/18810/Desktop/FL/net_global.pkl"
    save_dir = "./global_model/net_global.pkl"
    save_dir_node11 = "./ws-mnist-keras/node1/model/"
    save_dir_node12 = "./ws-mnist-keras/node2/model/"
    save_dir_node13 = "./ws-mnist-keras/node3/model/"
    save_dir_node21 = "./ws2-mnist-pytorch/node1/model/"
    save_dir_node22 = "./ws2-mnist-pytorch/node2/model/"
    save_dir_node23 = "./ws2-mnist-pytorch/node2/model/"
<<<<<<< HEAD
    count=0
    Effectweights=[1,1]
    Rubostweights=[math.log(3),math.log(3)]
    weights=[0,0]
    # time.sleep(3)
    while True:

        # check wether FDL start :
        while True:
            if GFDL_runnot_flag() == True:
                print("All Distributed learning is Finished")
                break
            else:
                print("Distributed Learning Still not Finished, wait for 3 second")
                time.sleep(3)
        # Global Model Aggregation
        model_1.load_state_dict(torch.load(read_dir_1 +"net_1.pkl"))
        model_2.load_state_dict(torch.load(read_dir_2+"net_1.pkl"))
        distance_error_1, loss_1, acc_1=test(model_1, device, testLoader, test_dataset, conf)
        distance_error_2, loss_2, acc_2=test(model_2, device, testLoader, test_dataset, conf)
        w_local = []
        w_local.append(copy.deepcopy(model_1.state_dict()))
        w_local.append(copy.deepcopy(model_2.state_dict()))
        # Weights Calculation and Prediction
        if float(loss_1)<float(loss_2):
            Effectweights[0] += 1
        else:
            Effectweights[1] += 1
        weights[0],weights[1]=Effectweights[0]+Rubostweights[0],Effectweights[1]+Rubostweights[1]
        # w_global = get_weights(w_local, [weights[0]/(weights[0]+weights[1]), weights[1]/(weights[0]+weights[1])])
        w_global = get_weights(w_local, [0.5, 0.5])
        model_global.load_state_dict(w_global)
        # Test Global Model, and write test data
        distance_error, loss, acc = test(model_global, device, testLoader, test_dataset, conf)
        f = open('global_acc.csv', 'a', encoding='utf-8')
        csv_writer = csv.writer(f)
        csv_writer.writerow([str(distance_error_1), str(loss_1),str(acc_1),str(distance_error_2), str(loss_2),str(acc_2),str(distance_error), str(loss),str(acc)])
        f.close()
        print("Finish FDL model generationg and save csv file")

        #Save net data
        torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node1/model/net_2.pkl")
        torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node2/model/net_2.pkl")
        torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node3/model/net_2.pkl")
        torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node1/model/net_2.pkl")
        torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node2/model/net_2.pkl")
        torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node3/model/net_2.pkl")
        print("save FDL models")
        # Tell FDL Stop
        GFDL_runnot_flag_change2True()
        print("change FDL signal to True and wait to start Distributed Learning")
        time.sleep(10)
    torch.save(model_global, save_dir)



        # # if True:
        # while True:
        #     if model_2_flag=="True" and model_1_flag=="True" :
        #         print("Distributed model 11 and 21 is generated")
        #         model_1.load_state_dict(torch.load(read_dir_1+"net.pkl"))
        #         model_2.load_state_dict(torch.load(read_dir_2+"net.pkl"))
        #         distance_error_1, loss_1, acc_1=test(model_1, device, testLoader, test_dataset, conf)
        #         distance_error_2, loss_2, acc_2=test(model_2, device, testLoader, test_dataset, conf)
        #         # if float(loss_1)<float(loss_2):
        #         #     weights[0] += 1
        #         # else:
        #         #     weights[1] += 1
        #         w_local = []
        #         w_local.append(copy.deepcopy(model_1.state_dict()))
        #         w_local.append(copy.deepcopy(model_2.state_dict()))
        #         # w_local.append(copy.deepcopy(model_2.state_dict()))
        #         # w_global = get_weights(w_local, [weights[0]/(weights[0]+weights[1]), weights[1]/(weights[0]+weights[1])])
        #         w_global = get_weights(w_local, [0.5,0.5])
        #
        #         print(weights)
        #
        #         model_global.load_state_dict(w_global)
        #         distance_error, loss, acc=test(model_global, device, testLoader, test_dataset, conf)
        #         f = open('global_acc.csv', 'a', encoding='utf-8')
        #         csv_writer = csv.writer(f)
        #         csv_writer.writerow([str(distance_error_1), str(loss_1),str(acc_1),str(distance_error_2), str(loss_2),str(acc_2),str(distance_error), str(loss),str(acc)])
        #         f.close()
        #         print("Finish FDL model generationg and save csv file")
        #         # torch.save(model_global.state_dict(), read_dir_1+"net.pkl")
        #         # torch.save(model_global.state_dict(), read_dir_2+"net.pkl")
        #         # torch.save(model_global.state_dict(), save_dir_node11+"net.pkl")
        #         # torch.save(model_global.state_dict(), save_dir_node12+"net.pkl")
        #         # torch.save(model_global.state_dict(), save_dir_node13+"net.pkl")
        #         # torch.save(model_global.state_dict(), save_dir_node21+"net.pkl")
        #         # torch.save(model_global.state_dict(), save_dir_node22+"net.pkl")
        #         # torch.save(model_global.state_dict(), save_dir_node23+"net.pkl")
        #         torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning_0917_withdiscription/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node1/model/FL_model11.net.pkl")
        #         torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning_0917_withdiscription/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node2/model/FL_model12.net.pkl")
        #         torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning_0917_withdiscription/IoV-SFDL-Swarm-Federated-Deep-Learning/ws-mnist-keras/node3/model/FL_model13.net.pkl")
        #         torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning_0917_withdiscription/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node1/model/FL_model21.net.pkl")
        #         torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning_0917_withdiscription/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node2/model/FL_model22.net.pkl")
        #         torch.save(model_global.state_dict(), "/home/tylor/satamnt/swarm-federated-deep-learning_0917_withdiscription/IoV-SFDL-Swarm-Federated-Deep-Learning/ws2-mnist-pytorch/node3/model/FL_model23.net.pkl")
        #         print("save FDL models")
        #
        #         count += 1
        #         # with open(read_dir_1 + "read_model.txt", "w") as f:
        #         #     model_1_flag = f.readline()
        #
        #         with open(save_dir_node11 + "localmodel_txt.txt", "w") as f:
        #             f.write("False")  # 自带文件关闭功能，不需要再写f.close()
        #         with open(save_dir_node12 + "localmodel_txt.txt", "w") as f:
        #             f.write("False")  # 自带文件关闭功能，不需要再写f.close()
        #         print("change distirbuted 11 and 21 txt file to False")
        #
        #         # 生成GFDL 的 信号：
        #         GFDL_runnot_flag_change2True()
        #         print("change FDL signal to True and wait to start Distributed Learning")
        #
        #         break
        #     else:
        #         print("node11 and node 21 not generated, wait 4 seconds")
        #         time.sleep(4)

            # with open(save_dir_node13 + "read_model.txt", "w") as f:
            #     f.write("True")  # 自带文件关闭功能，不需要再写f.close()
            #
            # with open(save_dir_node21 + "read_model.txt", "w") as f:
            #     f.write("True")  # 自带文件关闭功能，不需要再写f.close()
            #
            # with open(save_dir_node22 + "read_model.txt", "w") as f:
            #     f.write("True")  # 自带文件关闭功能，不需要再写f.close()
            #
            # with open(save_dir_node23 + "read_model.txt", "w") as f:
            #     f.write("True")  # 自带文件关闭功能，不需要再写f.close()

            # print("Save csv loss")
=======


    count=0
    weights=[1,1]
    time.sleep(15)
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
            w_global = get_weights(w_local, [weights[0]/(weights[0]+weights[1]), weights[1]/(weights[0]*0.5+weights[1]*0.5)])
            # w_global = get_weights(w_local, [0.5,0.5])

            model_global.load_state_dict(w_global)
            distance_error, loss, acc=test(model_global, device, testLoader, test_dataset, conf)
            f = open('global_acc.csv', 'a', encoding='utf-8')
            csv_writer = csv.writer(f)
            csv_writer.writerow([str(distance_error), str(loss),str(acc)])
            f.close()
            # torch.save(model_global.state_dict(), read_dir_1+"net.pkl")
            # torch.save(model_global.state_dict(), read_dir_2+"net.pkl")
            torch.save(model_global.state_dict(), save_dir_node11+"net.pkl")
            torch.save(model_global.state_dict(), save_dir_node12+"net.pkl")
            torch.save(model_global.state_dict(), save_dir_node13+"net.pkl")
            torch.save(model_global.state_dict(), save_dir_node21+"net.pkl")
            torch.save(model_global.state_dict(), save_dir_node22+"net.pkl")
            torch.save(model_global.state_dict(), save_dir_node23+"net.pkl")
            count += 1
            # with open(read_dir_1 + "read_model.txt", "w") as f:
            #     model_1_flag = f.readline()

            with open(save_dir_node11 + "read_model.txt", "w") as f:
                f.write("True")  # 自带文件关闭功能，不需要再写f.close()

            with open(save_dir_node12 + "read_model.txt", "w") as f:
                f.write("True")  # 自带文件关闭功能，不需要再写f.close()

            with open(save_dir_node13 + "read_model.txt", "w") as f:
                f.write("True")  # 自带文件关闭功能，不需要再写f.close()

            with open(save_dir_node21 + "read_model.txt", "w") as f:
                f.write("True")  # 自带文件关闭功能，不需要再写f.close()

            with open(save_dir_node22 + "read_model.txt", "w") as f:
                f.write("True")  # 自带文件关闭功能，不需要再写f.close()

            with open(save_dir_node23 + "read_model.txt", "w") as f:
                f.write("True")  # 自带文件关闭功能，不需要再写f.close()


>>>>>>> c570179daa53c7cd2cafe6e9b4d95446932832f4
            # with open(read_dir_1 + "read_model.txt", "w") as f:
            #     f.write("True")  # 自带文件关闭功能，不需要再写f.close()
            #
            # with open(read_dir_2 + "read_model.txt", "w") as f:
            #     f.write("True")  # 自带文件关闭功能，不需要再写f.close()
            # with open(read_dir_2 + "read_model.txt", "w") as f:
            #     model_2_flag = f.readline()
<<<<<<< HEAD
=======
        time.sleep(8)
>>>>>>> c570179daa53c7cd2cafe6e9b4d95446932832f4

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


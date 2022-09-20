import torch
import torch.nn.modules as nn
import numpy as np


class VPTLSTM(nn.Module):

    def __init__(self, rnn_size, embedding_size, input_size, output_size, grids_width, grids_height, dropout_par,
                 device):

        super(VPTLSTM, self).__init__()
        ######参数初始化##########
        self.device = device
        self.rnn_size = rnn_size  # hidden size默认128
        self.embedding_size = embedding_size  # 空间坐标嵌入尺寸64，每个状态用64维向量表示
        self.input_size = input_size  # 输入尺寸6,特征向量长度
        self.output_size = output_size  # 输出尺寸5
        self.grids_width = grids_width
        self.grids_height = grids_height
        self.dropout_par = dropout_par

        ############网络层初始化###############
        # 输入embeded_input,hidden_states
        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)

        # 输入Embed层，将长度为input_size的vec映射到embedding_size
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)

        # 输入[vehicle_num,grids_height,grids_width,rnn_size]  [26,39,5,128]
        # 输出[vehicle_num,grids_height-12,grids_width-4,rnn_size*4]  [26,27,1,32]
        self.social_tensor_conv1 = nn.Conv2d(in_channels=self.rnn_size, out_channels=self.rnn_size // 2, kernel_size=(5,3),
                                             stride=(2,1))
        self.social_tensor_conv2 = nn.Conv2d(in_channels=self.rnn_size // 2, out_channels=self.rnn_size // 4,
                                             kernel_size=(5,3), stride=1)
        self.social_tensor_embed = nn.Linear((self.grids_height - 15) * (self.grids_width - 4) * self.rnn_size // 4,
                                             self.embedding_size)

        # 输出Embed层，将长度为64的hidden_state映射到5
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_par)

    def forward(self, x_seq, grids, hidden_states, cell_states, long_term=False):
        '''
        模型前向传播
        params:
        x_seq: 输入的一组数据tensor(seq_len=99,vehicle_num=26,input_size=9)
        grids: 相关性判断矩阵tensor(99,26,39,5)
        hidden_states: 隐藏状态，tensor(vehicle_num=26,rnn_size=128)
        cell_states: 记忆胞元，tensor(vehicle_num=26,rnn_size=128)
        long_term:长时预测模式

        return:
        long_term=0:
        对应99个二维高斯函数[seq_length=99,vehicle_num=26,output_size=5]
        long_term ！=0:
        未来5秒预测
        '''
        self.x_seq = x_seq  # [seq_len=99,vehicle_num=26,input_size=9]
        self.grids = grids  # [seq_len=99,vehicle_num=26,grids_height=39,grid_width=5]
        self.hidden_states = hidden_states  # [vehicle_num=26,rnn_size=128]
        self.cell_states = cell_states  # [vehicle_num=26,rnn_size=128]

        if not long_term:
            outputs = []
            for frame_index, frame in enumerate(self.x_seq):
                output = self.frameForward(frame, grid=self.grids[frame_index])

                outputs.append(output)
            return torch.stack(outputs, dim=0)
        else:
            outputs = []
            last_point = None
            for frame_index, frame in enumerate(self.x_seq):
                last_point = self.frameForward(frame, grid=self.grids[frame_index])
            stable_value = self.x_seq[0, :, 2:7]
            for _ in range(self.x_seq.shape[0]):
                outputs.append(last_point.clone()) ##last_point是在外边的！tensor可以在循环外被保存
                last_point, grid = self.dataMakeUp(stable_value=stable_value, last_point=last_point)
                last_point = self.frameForward(last_point, grid=grid)

            return torch.stack(outputs)

    def frameForward(self, frame, grid):
        '''
        一帧正向传播，更新hidden_state和cell_state,返回下一个点预测结果
        输入：frame：tensor(vehicle_num=26,vec=9)
        输出：output：tensor(vehicle_num=26,vec=5)
        vec=[mx,my,sx,sy,corr]
        '''
        # 得到social_tensor:  tensor(vehicle_num=26,girds_height=39,grids_width=5,rnn_size=128)
        social_tensor = self.getSocialTensor(grid)

        # Embed inputs
        # 输入(vehicle_num,input_size),输出(vehicle_num,embedding_size=64)
        input_embedded = self.dropout(self.relu(self.input_embedding_layer(frame)))

        # Social_tensor的运算
        # 输入tensor(vehicle_num=26,rnn_size=128,girds_height=39,grids_width=5)，
        # 输出tensor(vehicle_num=26,rnn_size=128/2,girds_height=39-6,grids_width=5-2)
        social_tensor = social_tensor.permute(0, 3, 1, 2)
        tensor_embedded = self.dropout(self.relu(self.social_tensor_conv1(social_tensor)))

        # 输入tensor(vehicle_num=26,rnn_size=128/2,girds_height=39-6,grids_width=5-2)
        # 输出tensor(vehicle_num=26,rnn_size=128/4,girds_height=39-12,grids_width=5-4)
        tensor_embedded = self.dropout(self.relu(self.social_tensor_conv2(tensor_embedded)))

        # 输入tensor(vehicle_num=26,rnn_size=128/4,girds_height=39-12,grids_width=5-4)
        # 打平到tensor(vehicle_num=26,-1)
        # 全连接得到tensor(26,embeding_size)
        tensor_embedded = self.dropout(self.relu(self.social_tensor_embed(torch.flatten(tensor_embedded, 1))))

        # 拼接embed后的input和social_tensor  #输入2个(vehicle_num,embedding_size=64)输出(vehicle_num,2*embedding_size)
        concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

        # LSTM运行一次
        # 输入(vehicle_num,2*embedding_size),(2，[vehicle_num,rnn_size]),输出[2，[vehicle_num,rnn_size]]
        self.hidden_states, self.cell_states = self.cell(concat_embedded, (self.hidden_states, self.cell_states))

        # 计算下一帧output
        # 输入[vehicle_num,rnn_size],输出[vehicle_num,output_size]
        # output=self.sigmoid(self.output_layer(self.hidden_states))
        output = self.output_layer(self.hidden_states)

        return output

    def getSocialTensor(self, one_frame_grids):
        '''

        :param one_frame_grids: 一帧的相关性判断矩阵tensor(vehicle_num=26,grids_height=39,grids_width=5)，没车默认-1
        :return: social_tensor:嵌入相应隐藏张量的状态量tensor(vehicle_num=26,girds_height=39,grids_width=5，rnn_size=128)
        '''
        # 得到一个全为0的空的tensor(vehicle_num=26,grids_height=39,grids_width=5,rnn_size=128)
        social_tensor = torch.zeros_like(one_frame_grids.unsqueeze(-1).expand(-1, -1, -1, self.rnn_size))

        grid_have_vehicle = torch.where(one_frame_grids != -1)  # 收到3个等长tensor，3对应三个维度的索引值
        total_grids = grid_have_vehicle[0].shape[0]  # 总共有多少个有车的grid
        for one_grid in range(total_grids):
            # 三个维度
            target_vehicle_index, grids_height_index, grids_width_index = grid_have_vehicle[0][one_grid], \
                                                                          grid_have_vehicle[1][one_grid], \
                                                                          grid_have_vehicle[2][one_grid]
            social_tensor[target_vehicle_index][grids_height_index][grids_width_index] = self.hidden_states[
                target_vehicle_index]

        return social_tensor

    def getFunction(self, getGrid, road_info, min_Local_Y, max_Local_Y):
        self.getGrid = getGrid
        self.road_info = road_info
        self.min_Local_Y = min_Local_Y
        self.max_Local_Y = max_Local_Y

    def dataMakeUp(self, stable_value, last_point):
        '''
        9个特征[local_x, local_y, v_length, v_width, motor, auto, truck, turn_left, turn_right]
        :param stable_value: tensor(vehicle,5)固有属性v_length, v_width, motor, auto, truck
        :param last_point:上一个循环的tensor(vehicle_num,5)
        :return:
            combine_data: tensor[vehicle_num,vec=9]  补齐的数据
            grid:tensor(vehicle,grid_height,grid_width)
        '''
        combine_data = torch.cat([last_point[:, 0:2], stable_value], dim=1)
        turn_left = torch.as_tensor(combine_data[:, 0] * self.road_info["max_Local_X"] > self.road_info["lane_one_max"],
                                    dtype=torch.int, device=self.device)
        turn_right = torch.as_tensor(combine_data[:, 0] * self.road_info["max_Local_X"] < self.road_info["lane_five_min"], dtype=torch.int, device=self.device)
        combine_data = torch.cat([combine_data, torch.unsqueeze(turn_left, dim=-1).float(), torch.unsqueeze(turn_right, dim=-1).float()], dim=1)

        last_point[:, 0] = last_point[:, 0] * self.road_info["max_Local_X"]
        last_point[:, 1] = last_point[:, 1] * (self.max_Local_Y - self.min_Local_Y) + self.min_Local_Y
        grid = self.getGrid(last_point, from_df=0)
        grid = torch.tensor(np.array(grid), device=self.device, dtype=torch.float32)
        return combine_data, grid

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import itertools


class myDataSet(Dataset):
    '''
    x,y的筛选，grids的计算
    输入见__init__
    long_term=False:
        输出x:tensor(seq_length-1,vehicle_num,9)
            9个特征[local_x,local_y,v_length,v_width,motor,auto,truck,turn_left,turn_right]
        输出y:tensor(seq_length-1,vehicle_num,2)
        输出grids:tensor(seq_len-1,vehicle_num,grids_height,grids_width)
    long_term=True:
        输出x:tensor(seq_length/2,vehicle_num,9)
            9个特征[local_x,local_y,v_length,v_width,motor,auto,truck,turn_left,turn_right]
        输出y:tensor(seq_length/2,vehicle_num,2)
        输出grids:tensor(seq_len=99,vehicle_num,grids_height,grids_width)
    '''

    def __init__(self, csv_source, need_col, output_col, grids_width, grids_height, meter_per_grid, road,
                 long_term=False):
        '''

        init用于得到[19,100,df(26,7)]，初始化所需参数
        :param csv_source: 已处理好csv文件位置batches*seq_length*vehicle_num行
        :param need_col: 所需列名称
        :param output_col: 最终输出包含列
        :param grids_width: 横向格子数
        :param grids_height: 纵向格子数
        :param meter_per_grid: 格子比例尺
        :param road: 路面名称
        :param long_term: 提取长时预测数据
        '''
        # 初始化参数
        self.output_col = output_col
        self.col_seq = dict(zip(output_col, [i for i in range(len(output_col))]))

        self.grids_width = grids_width
        self.grids_height = grids_height
        self.meter_per_grid = meter_per_grid

        self.long_term = long_term

        self.road_info = {"US101_info": {"min_Local_X": 0, "max_Local_X": 24, "max_Lane_ID": 5, "min_v_length": 1.2,
                                         "max_v_length": 23.2, "min_v_Width": 0.6, "max_v_Width": 2.6,
                                         "max_Local_Y": 682, "lane_one_max": 4.1, "lane_five_min": 13}}
        self.road_info = self.road_info[road]
        # 开始初始化
        self.init_data = pd.read_csv(csv_source,
                                     usecols=need_col)  # [batches*seq_length*vehicle_num+batches, len(need_col)]
        self.zip_data = self.cutbyDelimiter(self.init_data)
        self.zip_data = self.cutTofinal(self.zip_data)
        print("{}文件加载完成，共计{}组数据".format(csv_source, len(self.zip_data)))

    def cutTofinal(self, zip_data):
        '''

        :param zip_data: [19, df(2600,8)]
        :return: [19,100,df(26,8)]
        '''
        for seq, one_zip_data in enumerate(zip_data):
            time_groups = one_zip_data["Global_Time"].unique()
            time_zip_data = one_zip_data.groupby("Global_Time")
            time_data = []  # 得到[100，（26，8）]

            for time in time_groups:
                one_time_data = time_zip_data.get_group(time)  # （26，8）
                time_data.append(one_time_data)
            zip_data[seq] = time_data

        return zip_data

    def cutbyDelimiter(self, init_data):
        '''

        :param init_data: 初始的带有分割行的df
        :return: 分割后的df，在同一个list
        '''
        _index = 0
        zip_data = []
        for index, row in init_data.iterrows():
            if row['Vehicle_ID'] == "Vehicle_ID":
                zip_data.append(init_data[_index:index])
                _index = index + 1
        return zip_data

    def __getitem__(self, item):
        '''
        继续init的步骤，从提取的[batchs,seq_length,df(26,8)]中输出数据
        筛选了19组数据，每组数据包含100个时间点，第一组数据中有26辆车，共输入8个特征
        提取数据，[num_batch,df(seq_length*vehicle_num,len(need_col)]
        long_term=False:
            输出x:tensor(seq_length-1,vehicle_num,9)
                9个特征[local_x,local_y,v_length,v_width,motor,auto,truck,turn_left,turn_right]
            输出y:tensor(seq_length-1,vehicle_num,2)
            输出grids:tensor(seq_len-1,vehicle_num,grids_height,grids_width)
        long_term=True:
            输出x:tensor(seq_length/2,vehicle_num,9)
                9个特征[local_x,local_y,v_length,v_width,motor,auto,truck,turn_left,turn_right]
            输出y:tensor(seq_length/2,vehicle_num,2)
            输出grids:tensor(seq_len=99,vehicle_num,grids_height,grids_width)
        '''
        grids = []
        seq_data = []
        # zip_data[item]： [seq_length=100,df(vehicle_num=26,vec=8)]  one_frame_df:df(vehicle_num=26,vec=8)
        for one_frame_df in self.zip_data[item]:
            one_frame_df = one_frame_df.loc[:, self.output_col]  # df(vehicle_num=26,vec=6)
            grid = self.getGrid(one_frame_df)
            grids.append(grid)
            seq_data.append(one_frame_df.values.astype(float))

        # 非长时预测
        if not self.long_term:
            grids = np.array(grids[:-1])  # [seq_length-1,grids_height,grids_width]

            seq_data = np.array(seq_data)
            seq_data = self.normalization(seq_data=seq_data, col_seq=self.col_seq)

            x_seq_data = seq_data[:-1]  # [seq_length-1,df(vehicle_num=26,vec=6)]
            y_seq_data = seq_data[1:, :,
                         self.col_seq['Local_X']:self.col_seq['Local_Y'] + 1]  # [seq_length-1,df(vehicle_num=26,vec=2)]

            return x_seq_data, y_seq_data, grids

        # 长时预测
        else:
            grids = np.array(grids)[0:len(grids) // 2]  # [seq_length/2,grids_height,grids_width]

            seq_data = np.array(seq_data)
            seq_data = self.normalization(seq_data=seq_data, col_seq=self.col_seq)

            x_seq_data = seq_data[:seq_data.shape[0] // 2]  # [seq_length/2,df(vehicle_num=26,vec=6)]
            y_seq_data = seq_data[seq_data.shape[0] // 2:, :,
                         self.col_seq['Local_X']:self.col_seq['Local_Y'] + 1]  # [seq_length/2,df(vehicle_num=26,vec=2)]

            return x_seq_data, y_seq_data, grids

    def normalization(self, seq_data, col_seq):  # [100,26,6]
        '''
        归一化x_seq_data
        :param seq_data: array(seq_length=100,vehicle_num=26,vec=8)
                         vec=("Local_X","Local_Y","v_length","v_Width","v_Class","Lane_ID")
        :param col_seq: 列的排列顺序
        :param min_Local_Y: 最小y
        :param max_Local_Y: 最大y
        :return: seq_data array(seq_length=99,vehicle_num=26,vec=9)
                 vec=[local_x,local_y,v_length,v_width,motor,auto,truck,turn_left,turn_right]
        '''
        # local_X归一化
        seq_data[:, :, col_seq['Local_X']] = seq_data[:, :, col_seq['Local_X']] / (self.road_info["max_Local_X"])

        # local_Y归一化
        self.min_Local_Y, self.max_Local_Y = np.min(seq_data[:, :, col_seq['Local_Y']]), np.max(
            seq_data[:, :, col_seq['Local_Y']]) + 20
        seq_data[:, :, col_seq['Local_Y']] = (seq_data[:, :, col_seq['Local_Y']] - self.min_Local_Y) / (
                self.max_Local_Y - self.min_Local_Y)
        # seq_data[:, :, col_seq['Local_Y']] = (seq_data[:, :,col_seq['Local_Y']]) /self.road_info["max_Local_Y"]

        # v_length归一化
        seq_data[:, :, col_seq["v_length"]] = (seq_data[:, :, col_seq["v_length"]] - self.road_info[
            "min_v_length"]) / (self.road_info["max_v_length"] - self.road_info["min_v_length"])

        # v_width归一化
        seq_data[:, :, col_seq["v_Width"]] = (seq_data[:, :, col_seq["v_Width"]] - self.road_info[
            "min_v_Width"]) / (self.road_info["max_v_Width"] - self.road_info["min_v_Width"])

        # v_Class归一化
        v_Class = seq_data[:, :, col_seq["v_Class"]]
        v_Class = self.oneHot_v_Class(v_Class)
        seq_data = np.concatenate([seq_data, v_Class], axis=2)  # motor,auto,truck

        # Lane_ID归一化
        Land_ID = seq_data[:, :, col_seq["Lane_ID"]]
        Land_ID = self.exeLane_ID(Land_ID)
        seq_data = np.concatenate([seq_data, Land_ID], axis=2)
        seq_data = np.delete(seq_data, [col_seq["v_Class"], col_seq["Lane_ID"]], axis=2)
        return seq_data

    def exeLane_ID(self, data):
        '''
        处理车道 二维数组
        输入：narray[seq_length, vehicle_ID,1]
        输出：narray[seq_length, vehicle_ID,2]
        '''
        exeLane_ID_data = []
        for frame in data:
            vehicle_data = []
            for one_vehicle in frame:
                if one_vehicle == 1:
                    vehicle_data.append([0, 1])
                elif one_vehicle == 5:
                    vehicle_data.append([1, 0])
                else:
                    vehicle_data.append([1, 1])
            exeLane_ID_data.append(vehicle_data)
        return np.array(exeLane_ID_data)

    def oneHot_v_Class(self, data):
        '''
        One hot 二维数组
        输入：narray[seq_length, vehicle_ID,1]
        输出：narray[seq_length, vehicle_ID,3]
        '''
        one_hot_data = []
        for frame in data:
            one_hot_data.append(np.eye(3)[np.array(frame - 1, dtype=np.int)])
        return np.array(one_hot_data, dtype=np.int)

    def getGrid(self, x_seq, from_df=1):
        '''
        亮瞎眼金坷垃闪光BUFF：为每一帧，每一辆车作为目标车，生成一个[grids_width,grids_height]的网格
        :param x_seq: df(vehicle_num=26,input_size=6)，只需要input_size的前两项，即Local_x，Local_y
        :param grids_width:横向格子数,奇数！
        :param grids_height:纵向格子数，奇数！
        :param meter_per_grid:格子边长代表的米数
        :param from_df:长时运算传入的非df

        :return:[vehicle_num=26,grids_height,grids_width]用以过滤hidden_state
        '''
        center_width = int(self.grids_width / 2)  # 对于5就是2  [0,1,2,3,4]
        center_height = int(self.grids_height / 2)  # 对于39就是19

        vehicle_num = x_seq.shape[0]

        masks = np.zeros((vehicle_num, self.grids_height, self.grids_width)) - 1  # | 土 |  (39,5)

        if from_df:
            x_seq = x_seq.values.astype(float)

        for vehicle_a, vehicle_b in itertools.combinations(list(range(vehicle_num)), 2):

            width_dist = int((x_seq[vehicle_a][self.col_seq["Local_X"]] - x_seq[vehicle_b][
                self.col_seq["Local_X"]]) / self.meter_per_grid)  # 以b作为目标车时
            height_dist = int((x_seq[vehicle_a][self.col_seq["Local_Y"]] - x_seq[vehicle_b][
                self.col_seq["Local_Y"]]) / self.meter_per_grid)

            if abs(width_dist) > center_width or abs(height_dist) > center_height:
                continue
            else:
                # 越往右侧x索引越大，越往上侧y索引越小
                masks[vehicle_b][center_height - height_dist][center_width + width_dist] = vehicle_a
                masks[vehicle_a][center_height + height_dist][center_width - width_dist] = vehicle_b
        return masks.astype(np.int)

    def __len__(self):
        return len(self.zip_data)

# import pdb
import random

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from utils.hand.enhance import finger_tapping_distance
from utils.seed import set_seed


class PDHandData(Dataset):
    # start from 0.5s (30~)
    # random cut interest region: 5s (300 points)
    def __init__(self,
                 filename_label_df,
                 input_channels=[],
                 data_root="../Data/Hand/dataset_openpose/left_1",
                 mk_balanced_dataset=False,
                 mk_balanced_type="oversampling",
                 multi_sample_type="replicate",
                 multi_sample_num: int = 1,
                 random_rotat_3d: bool = False,
                 seed=42,
                 return_label=True,
                 return_name=False,
                 enhanced_type: int = 0,  # 0: None; 1: finger tapping; 2: open/closing; 3: supination/pronation
                 group_map: dict = {"HC": 0, "PD": 1, "MSA": 2, "other": 3},
                 gaussian_sampling=False,
                 crop_len=300           # 7s:420; 5s: 300
                 ):
        '''
        filename_label_df: [0]: filename, [1]: label
        '''

        self.seed = seed
        set_seed(self.seed)

        filename_label_df.reset_index(inplace=True, drop=True)
        self.filename_label_df = filename_label_df[[0, 1]]
        self.filename_list = list(self.filename_label_df[0])
        self.label_list = list(self.filename_label_df[1])
        self.gaussian_sampling = gaussian_sampling
        self.enhanced_type = int(enhanced_type)
        self.input_channels = input_channels
        self.random_rotat_3d = random_rotat_3d

        self.data_root = data_root
        self.return_label = return_label
        self.return_name = return_name

        if self.gaussian_sampling:
            self.random_list = np.clip(np.random.normal(loc=0.5, scale=0.12, size=len(self.label_list)), 0, 1)
        else:
            self.random_list = np.random.rand(len(self.label_list))

        # self.outlier_criteria = 0.1
        self.crop_len = crop_len
        self.group_map = group_map

        if mk_balanced_dataset:
            if mk_balanced_type == "random-crop":
                # due to the random cropping process in "reading_hand_csv_pipeline",
                # different input dataset can be created.
                # Even if some input data are from the same recording, the dataset can enlarge and make balance between groups.
                # Here, it is necessary to make some replicates in "filename_list" and "label_list" (should be paired) for less recording data.
                self._random_crop_sampling()

            elif mk_balanced_type == "oversampling":
                # in oversampling method, different from balanced method, the random start will not change with sampling
                # it is just replicate data in the smaller group to make the number of data balanced.
                self._oversampling()

            else:
                raise NotImplementedError

        # make some replicates in "filename_list" and "label_list" (should be paired)
        # to enlarge dataset based on the random cropping process in "reading_hand_csv_pipeline"
        if multi_sample_type == "random-crop":
            self.filename_list = self.filename_list * multi_sample_num
            self.label_list = self.label_list * multi_sample_num
            if self.gaussian_sampling:
                # gaussian random
                self.random_list = list(self.random_list) + list(np.clip(np.random.normal(loc=0.5, scale=0.12, size=len(self.label_list) - len(self.random_list)), 0, 1))
            else:
                # uniform random
                self.random_list = list(self.random_list) + list(np.random.rand(len(self.label_list) - len(self.random_list)))
        elif multi_sample_type == "replicate":
            self.filename_list = self.filename_list * multi_sample_num
            self.label_list = self.label_list * multi_sample_num
            self.random_list = list(self.random_list) * multi_sample_num

        else:
            raise NotImplementedError

    # to process hand csv
    def read_hand_csv_pipeline(self, csv_name, idx):
        data = self._read_hand_csv(csv_name)
        # data = self._cut_outlier(data) #cant use when timestamp are included
        data = self._timestamp_normalization(data)
        data = self._padding(data)
        data = self._cropping(data, idx)
        if self.random_rotat_3d:
            data = self._random_rotat_3d(data)
        return data

    # necessary part of the dataset class
    def __len__(self):
        return len(self.filename_list)

    # necessary part of the dataset class
    def __getitem__(self, idx):
        csv_name = self.filename_list[idx]
        data = np.float32(self.read_hand_csv_pipeline(csv_name, idx))

        if self.return_label:
            label = self.group_map[self.label_list[idx]]
            if self.return_name:
                return data, label, csv_name
            else:
                return data, label
        else:
            if self.return_name:
                return data, csv_name
            else:
                return data

    # other required functions
    def _random_crop_sampling(self):
        # count the number of data in each group
        group_count = self.filename_label_df.groupby([1]).agg('count')

        group_dict = {
            label: group_count.iloc[i].item()
            for i, label in enumerate(group_count.index)
        }
        max_num = group_count.max().item()

        # random sample to enlarge list of filename and label
        # the position of starting point is created, either (the starting point would be randomly sampled)
        for label in group_dict.keys():
            sampling_pool = self.filename_label_df[self.filename_label_df[1] == label]
            orginal_num = group_dict[label]
            supply_num = max_num - orginal_num

            supply_filename_label_df = sampling_pool.sample(
                supply_num, replace=True, random_state=self.seed)

            self.filename_list = self.filename_list + list(
                supply_filename_label_df[0])
            self.label_list = self.label_list + list(
                supply_filename_label_df[1])

        if self.gaussian_sampling:
            self.random_list = np.clip(np.random.normal(loc=0.5, scale=0.12, size=len(self.label_list)), 0, 1)
        else:
            self.random_list = np.random.rand(len(self.label_list))
        return None

    def _oversampling(self):
        # the position of starting point is created first (the starting point would be randomly sampled)
        # then, random sample to enlarge list of filename and label

        # define the starting point of each file
        filename_label_rdnstart = pd.DataFrame([
            self.filename_list,
            self.label_list,
            self.random_list])
        filename_label_rdnstart = filename_label_rdnstart.transpose()
        group_count = filename_label_rdnstart[[0, 1]].groupby([1]).agg('count')
        group_dict = {
            label: group_count.iloc[i].item()
            for i, label in enumerate(group_count.index)
        }
        max_num = group_count.max().item()

        # oversampling
        for label in group_dict.keys():
            sampling_pool = filename_label_rdnstart[filename_label_rdnstart[1] == label]
            orginal_num = group_dict[label]
            supply_num = max_num - orginal_num

            supply_filename_label_df = sampling_pool.sample(
                supply_num, replace=True, random_state=self.seed)

            self.filename_list = self.filename_list + list(
                supply_filename_label_df[0])
            self.label_list = self.label_list + list(
                supply_filename_label_df[1])
            self.random_list = list(self.random_list) + list(
                supply_filename_label_df[2]
            )

        return None

    def _read_hand_csv(self, csv_name):
        # read from file
        csv_path = f"{self.data_root}/{csv_name}"
        data = pd.read_csv(csv_path)

        # enhance features and reshape data
        if self.enhanced_type == 0:
            # print("No enhancement techniques are included.")
            pass
        elif self.enhanced_type == 1:
            try:
                data['enhanced_feat'] = finger_tapping_distance(data, kpt_method="3D")
            except Exception:
                data['enhanced_feat'] = finger_tapping_distance(data, kpt_method="2D")
        elif self.enhanced_type == 2:
            pass
        elif self.enhanced_type == 3:
            pass
        else:
            raise NotImplementedError

        # reshape
        data = data[self.input_channels].values  # L,C
        data = data.T  # L, C-> C, L for pytorch input

        return data

    def _timestamp_normalization(self, data):
        data[0, :] = data[0, :] / 60 / 1000  # "frames" are transformed to "ks"
        return data

    # def _cut_outlier(self, data):
    #     data[np.abs(data) > self.outlier_criteria] = 0
    #     return data

    def _random_rotat_3d(self, data, rnd_theta=90):
        # rotation setting
        rotation_order = random.choice([
            "zxz",
            "xyx",
            "yzy",
            "zyz",
            "xzx",
            "yxy",  # Proper Euler angles
            "xyz",
            "yzx",
            "zxy",
            "xzy",
            "zyx",
            "yxz"
        ])  # Tait–Bryan angles
        rnd_angle_one = random.randint(0, rnd_theta)
        rnd_angle_two = random.randint(0, rnd_theta)
        rnd_angle_three = random.randint(0, rnd_theta)

        r = R.from_euler(rotation_order,
                         [rnd_angle_one, rnd_angle_two, rnd_angle_three],
                         degrees=True)

        # data processing
        data = pd.DataFrame(data.T, columns=self.input_channels)
        axis_num = pd.Series([each.split("_")[0] for each in self.input_channels if ("_" in each) and not ("enhanced_feat" in each)]).unique()
        kpts_num = pd.Series([each.split("_")[-1] for each in self.input_channels if ("_" in each) and not ("enhanced_feat" in each)]).unique()
        assert len(axis_num) == 3, "please include 3D keypoints"

        new_data_list = []
        for num in kpts_num:
            data_each_kpts = data[[f'x_{num}', f'y_{num}', f'z_{num}']]
            new_data = np.dot(r.as_matrix(), data_each_kpts.values.T)
            new_data_list.append(new_data)

        new_data = np.array(new_data_list)
        # new_data: (kpts_num, axis_num[x,y,z], timeframe)
        new_data = new_data.transpose((1, 0, 2))
        # new_data: (axis_num[x,y,z], kpts_num, timeframe)
        new_data = new_data.reshape((len(axis_num) * len(kpts_num), self.crop_len))
        # new_data: ([x1,x2,x3,...,y1,y2,y3,...,z1,z2,z3,...], timeframe)
        new_data = new_data.T
        # new_data: (timeframe, all_kpts)
        new_data = np.insert(new_data, 0, data['timestamp'].values, axis=1)
        # new_data: (timeframe, timestamp+all_kpts)
        new_data = new_data.T
        # new_data: (all_channels, timeframe)
        if "enhanced_feat" in self.input_channels:
            new_data = np.concatenate([new_data, [data['enhanced_feat'].values]], axis=0)

        return new_data

    def _padding(self, data):
        if data.shape[1] < self.crop_len:
            padding_len = self.crop_len - data.shape[1]
            data = np.pad(data, ((0, 0), (0, padding_len)),
                          'constant',
                          constant_values=0)
        return data

    def _cropping(self, data, idx):
        # random_start = np.random.randint(
        #     0, (data.shape[1] - self.crop_len + 1))
        random_start = int(np.floor((data.shape[1] - self.crop_len + 1) * self.random_list[idx]))
        return data[:, random_start:random_start + self.crop_len]


####

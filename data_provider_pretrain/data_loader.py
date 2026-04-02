import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

def _resolve_window_sizes(size):
    if size is None:
        raise ValueError('Explicit [seq_len, label_len, pred_len] is required for dataset construction.')
    return int(size[0]), int(size[1]), int(size[2])


def _build_time_features(df_stamp, timeenc, freq, include_minute=False):
    stamp = df_stamp[['date']].copy()
    stamp['date'] = pd.to_datetime(stamp['date'])
    if timeenc == 0:
        stamp['month'] = stamp.date.apply(lambda row: row.month, 1)
        stamp['day'] = stamp.date.apply(lambda row: row.day, 1)
        stamp['weekday'] = stamp.date.apply(lambda row: row.weekday(), 1)
        stamp['hour'] = stamp.date.apply(lambda row: row.hour, 1)
        if include_minute:
            stamp['minute'] = stamp.date.apply(lambda row: row.minute, 1)
            stamp['minute'] = stamp.minute.map(lambda x: x // 15)
        return stamp.drop(columns=['date']).values
    data_stamp = time_features(pd.to_datetime(stamp['date'].values), freq=freq)
    return data_stamp.transpose(1, 0)


def _validate_non_negative_borders(seq_len, border1s, border2s, context_name):
    if min(border1s) < 0:
        raise ValueError(
            f'{context_name} produced a negative split boundary with seq_len={seq_len}. '
            'Reduce seq_len or increase the split window.'
        )
    if any(b2 <= b1 for b1, b2 in zip(border1s, border2s)):
        raise ValueError(f'{context_name} produced invalid split windows: border1s={border1s}, border2s={border2s}')


def _compute_ratio_boundaries(total_len, seq_len, train_ratio, val_ratio, test_ratio, context_name):
    total = train_ratio + val_ratio + test_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0 or abs(total - 1.0) > 1e-6:
        raise ValueError(
            f'{context_name} ratios must all be positive and sum to 1.0; '
            f'got train={train_ratio}, val={val_ratio}, test={test_ratio}.'
        )
    num_train = int(total_len * train_ratio)
    num_vali = int(total_len * val_ratio)
    num_test = total_len - num_train - num_vali
    if min(num_train, num_vali, num_test) <= 0:
        raise ValueError(
            f'{context_name} ratios produced invalid split sizes: '
            f'train={num_train}, val={num_vali}, test={num_test}, total={total_len}.'
        )
    border1s = [0, num_train - seq_len, total_len - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, total_len]
    _validate_non_negative_borders(seq_len, border1s, border2s, context_name)
    return border1s, border2s


def _compute_pretrain_boundaries(total_len, seq_len, train_ratio, eval_ratio, context_name):
    if train_ratio <= 0 or eval_ratio <= 0 or abs((train_ratio + eval_ratio) - 1.0) > 1e-6:
        raise ValueError(
            f'{context_name} ratios must be positive and sum to 1.0; '
            f'got train={train_ratio}, eval={eval_ratio}.'
        )
    train_points = int(total_len * train_ratio)
    eval_points = total_len - train_points
    if min(train_points, eval_points) <= 0:
        raise ValueError(f'{context_name} produced invalid split sizes for total={total_len}.')
    border1s = [0, train_points - seq_len, train_points - seq_len]
    border2s = [train_points, train_points + eval_points, train_points + eval_points]
    _validate_non_negative_borders(seq_len, border1s, border2s, context_name)
    return border1s, border2s


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, pretrain=True, channel_independence=0,
                 train_split_ratio=0.7, val_split_ratio=0.1, test_split_ratio=0.2,
                 pretrain_train_split_ratio=0.8, pretrain_eval_split_ratio=0.2, **kwargs):
        self.seq_len, self.label_len, self.pred_len = _resolve_window_sizes(size)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.pretrain = pretrain
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.channel_independence = bool(channel_independence)
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.pretrain_train_split_ratio = pretrain_train_split_ratio
        self.pretrain_eval_split_ratio = pretrain_eval_split_ratio

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.pretrain:
            border1s, border2s = _compute_pretrain_boundaries(
                len(df_raw), self.seq_len, self.pretrain_train_split_ratio, self.pretrain_eval_split_ratio, 'Pretrain ETT hour split'
            )
        else:
            border1s, border2s = _compute_ratio_boundaries(
                len(df_raw), self.seq_len, self.train_split_ratio, self.val_split_ratio, self.test_split_ratio, 'ETT hour split'
            )

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f'Unsupported features mode: {self.features}')

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            train_values = pd.DataFrame(train_data).to_numpy(dtype='float32')
            df_values = pd.DataFrame(df_data).to_numpy(dtype='float32')
            self.scaler.fit(train_values)
            data = self.scaler.transform(df_values)
        else:
            data = pd.DataFrame(df_data).to_numpy(dtype='float32')

        data = pd.DataFrame(data).to_numpy(dtype='float32')

        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        data_stamp = _build_time_features(df_stamp, self.timeenc, self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.channel_independence:
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        base_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        return base_len * self.enc_in if self.channel_independence else base_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None, pretrain=True, channel_independence=0,
                 train_split_ratio=0.7, val_split_ratio=0.1, test_split_ratio=0.2,
                 pretrain_train_split_ratio=0.8, pretrain_eval_split_ratio=0.2, **kwargs):
        self.seq_len, self.label_len, self.pred_len = _resolve_window_sizes(size)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.pretrain = pretrain
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.channel_independence = bool(channel_independence)
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.pretrain_train_split_ratio = pretrain_train_split_ratio
        self.pretrain_eval_split_ratio = pretrain_eval_split_ratio

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.pretrain:
            border1s, border2s = _compute_pretrain_boundaries(
                len(df_raw), self.seq_len, self.pretrain_train_split_ratio, self.pretrain_eval_split_ratio, 'Pretrain ETT minute split'
            )
        else:
            border1s, border2s = _compute_ratio_boundaries(
                len(df_raw), self.seq_len, self.train_split_ratio, self.val_split_ratio, self.test_split_ratio, 'ETT minute split'
            )

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f'Unsupported features mode: {self.features}')

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            train_values = pd.DataFrame(train_data).to_numpy(dtype='float32')
            df_values = pd.DataFrame(df_data).to_numpy(dtype='float32')
            self.scaler.fit(train_values)
            data = self.scaler.transform(df_values)
        else:
            data = pd.DataFrame(df_data).to_numpy(dtype='float32')

        data = pd.DataFrame(data).to_numpy(dtype='float32')

        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        data_stamp = _build_time_features(df_stamp, self.timeenc, self.freq, include_minute=True)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.channel_independence:
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        base_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        return base_len * self.enc_in if self.channel_independence else base_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

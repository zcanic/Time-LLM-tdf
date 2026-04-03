import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings
from collections import Counter

warnings.filterwarnings('ignore')
from utils.tools import parse_column_spec

CUSTOM_DATE_COLUMN_CANDIDATES = ('date', 'datetime', 'timestamp', 'ds', 'Date', 'DATE', '日期', '时间戳')


def _resolve_window_sizes(size):
    if size is None:
        raise ValueError('Explicit [seq_len, label_len, pred_len] is required for dataset construction.')
    return int(size[0]), int(size[1]), int(size[2])


def _resolve_date_column(df_raw, explicit_date_col=''):
    if explicit_date_col:
        if explicit_date_col not in df_raw.columns:
            raise ValueError(f"Custom date column '{explicit_date_col}' not found. Available columns: {list(df_raw.columns)}")
        return explicit_date_col

    for candidate in CUSTOM_DATE_COLUMN_CANDIDATES:
        if candidate in df_raw.columns:
            return candidate

    raise ValueError(
        'No recognized datetime column found. '
        f"Expected one of {CUSTOM_DATE_COLUMN_CANDIDATES} or pass --custom_date_col explicitly. Available columns: {list(df_raw.columns)}"
    )


def _validate_split_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError('Split ratios must all be positive.')
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f'Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio} (sum={total}).'
        )


def _validate_non_negative_borders(seq_len, border1s, border2s, context_name):
    if min(border1s) < 0:
        raise ValueError(
            f'{context_name} produced a negative split boundary with seq_len={seq_len}. '
            'Reduce seq_len or increase the split window.'
        )
    if any(b2 <= b1 for b1, b2 in zip(border1s, border2s)):
        raise ValueError(f'{context_name} produced invalid split windows: border1s={border1s}, border2s={border2s}')


def _compute_ratio_boundaries(total_len, seq_len, train_ratio, val_ratio, test_ratio, context_name):
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
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


def _compute_date_boundaries(df_raw, date_col, seq_len, train_end_date, val_end_date):
    train_cutoff = pd.to_datetime(train_end_date, errors='raise').to_pydatetime()
    val_cutoff = pd.to_datetime(val_end_date, errors='raise').to_pydatetime()
    if train_cutoff >= val_cutoff:
        raise ValueError('--train_end_date must be earlier than --val_end_date for custom datasets.')

    date_series = pd.to_datetime(df_raw[date_col], errors='raise')
    train_end = int((date_series <= train_cutoff).sum())
    val_end = int((date_series <= val_cutoff).sum())

    if train_end <= seq_len or val_end <= train_end or val_end >= len(df_raw):
        raise ValueError(
            'Date-based splits are invalid for the current dataset length/seq_len. '
            f'train_end={train_end}, val_end={val_end}, len={len(df_raw)}, seq_len={seq_len}.'
        )

    border1s = [0, train_end - seq_len, val_end - seq_len]
    border2s = [train_end, val_end, len(df_raw)]
    _validate_non_negative_borders(seq_len, border1s, border2s, 'Custom date-based split')
    return border1s, border2s


def _validate_requested_columns(df_raw, requested_cols, label):
    missing = [col for col in requested_cols if col not in df_raw.columns]
    if missing:
        raise ValueError(f'Missing {label} columns: {missing}. Available columns: {list(df_raw.columns)}')


def _safe_str(value):
    if pd.isna(value):
        return '未知'
    return str(value)


def _build_park_prompt(window_df, target_col):
    latest = window_df.iloc[-1]
    traffic_series = window_df['交通状况'].dropna().astype(str) if '交通状况' in window_df.columns else pd.Series(dtype=str)
    traffic_counter = Counter(traffic_series.tolist())
    traffic_summary = ', '.join(f'{key}:{count}' for key, count in traffic_counter.items()) if traffic_counter else '无'
    latest_traffic = _safe_str(latest.get('交通状况'))
    latest_environment = _safe_str(latest.get('环境描述'))
    weather_text = _safe_str(latest.get('weather_天气'))
    weekday = _safe_str(latest.get('holiday_星期'))
    date_tag = _safe_str(latest.get('holiday_日期标签'))
    holiday_name = _safe_str(latest.get('holiday_节假日名称'))
    is_weekend = _safe_str(latest.get('holiday_是否周末'))
    is_holiday = _safe_str(latest.get('holiday_是否节假日放假'))
    is_makeup = _safe_str(latest.get('holiday_是否调休上班'))
    temp_avg = _safe_str(latest.get('weather_平均气温_摄氏度'))
    temp_min = _safe_str(latest.get('weather_最低气温_摄氏度'))
    temp_max = _safe_str(latest.get('weather_最高气温_摄氏度'))
    precip = _safe_str(latest.get('weather_总降水量_毫米'))
    start_ts = _safe_str(window_df.iloc[0][window_df.columns[0]])
    end_ts = _safe_str(latest[window_df.columns[0]])
    latest_target = _safe_str(latest.get(target_col))

    return (
        f'Observed window from {start_ts} to {end_ts}. '
        f'Latest observed {target_col} is {latest_target}. '
        f'Latest traffic is {latest_traffic}; recent traffic counts: {traffic_summary}. '
        f'Latest environment description: {latest_environment}. '
        f'Latest weather: {weather_text}, average temperature {temp_avg}°C, '
        f'min {temp_min}°C, max {temp_max}°C, precipitation {precip} mm. '
        f'Calendar context: weekday {weekday}, date tag {date_tag}, weekend {is_weekend}, '
        f'holiday {is_holiday}, holiday name {holiday_name}, makeup workday {is_makeup}.'
    )


def _build_generic_prompt(window_df, target_col, prompt_cols):
    latest = window_df.iloc[-1]
    parts = [f'Latest observed {target_col}: {_safe_str(latest.get(target_col))}.']
    for col in prompt_cols:
        if col in latest.index:
            parts.append(f'{col}: {_safe_str(latest[col])}.')
    return ' '.join(parts)


def _build_time_features(df_stamp, date_col, timeenc, freq, include_minute=False):
    stamp = df_stamp[[date_col]].copy()
    stamp[date_col] = pd.to_datetime(stamp[date_col])
    if timeenc == 0:
        stamp['month'] = stamp[date_col].apply(lambda row: row.month, 1)
        stamp['day'] = stamp[date_col].apply(lambda row: row.day, 1)
        stamp['weekday'] = stamp[date_col].apply(lambda row: row.weekday(), 1)
        stamp['hour'] = stamp[date_col].apply(lambda row: row.hour, 1)
        if include_minute:
            stamp['minute'] = stamp[date_col].apply(lambda row: row.minute, 1)
            stamp['minute'] = stamp.minute.map(lambda x: x // 15)
        return stamp.drop(columns=[date_col]).values

    data_stamp = time_features(pd.to_datetime(stamp[date_col].values), freq=freq)
    return data_stamp.transpose(1, 0)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, channel_independence=0, train_split_ratio=0.7,
                 val_split_ratio=0.1, test_split_ratio=0.2, **kwargs):
        self.seq_len, self.label_len, self.pred_len = _resolve_window_sizes(size)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.channel_independence = bool(channel_independence)
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio

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
            train_values = np.asarray(train_data, dtype=np.float32)
            df_values = np.asarray(df_data, dtype=np.float32)
            self.scaler.fit(train_values)
            data = np.asarray(self.scaler.transform(df_values), dtype=np.float32)
        else:
            data = np.asarray(df_data, dtype=np.float32)

        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        data_stamp = _build_time_features(df_stamp, 'date', self.timeenc, self.freq)

        self.data_x = np.asarray(data[border1:border2], dtype=np.float32)
        self.data_y = np.asarray(data[border1:border2], dtype=np.float32)
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
                 seasonal_patterns=None, channel_independence=0, train_split_ratio=0.7,
                 val_split_ratio=0.1, test_split_ratio=0.2, **kwargs):
        self.seq_len, self.label_len, self.pred_len = _resolve_window_sizes(size)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.channel_independence = bool(channel_independence)
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

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
            train_values = np.asarray(train_data, dtype=np.float32)
            df_values = np.asarray(df_data, dtype=np.float32)
            self.scaler.fit(train_values)
            data = np.asarray(self.scaler.transform(df_values), dtype=np.float32)
        else:
            data = np.asarray(df_data, dtype=np.float32)

        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        data_stamp = _build_time_features(df_stamp, 'date', self.timeenc, self.freq, include_minute=True)

        self.data_x = np.asarray(data[border1:border2], dtype=np.float32)
        self.data_y = np.asarray(data[border1:border2], dtype=np.float32)
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


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, train_split_ratio=0.7, val_split_ratio=0.1, test_split_ratio=0.2,
                 train_end_date='', val_end_date='', custom_date_col='', channel_independence=0,
                 numeric_feature_cols='', prompt_context_cols='', dropna_feature_cols='', **kwargs):
        self.seq_len, self.label_len, self.pred_len = _resolve_window_sizes(size)
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.train_end_date = train_end_date
        self.val_end_date = val_end_date
        self.custom_date_col = custom_date_col
        self.channel_independence = bool(channel_independence)
        self.numeric_feature_cols = parse_column_spec(numeric_feature_cols)
        self.prompt_context_cols = parse_column_spec(prompt_context_cols)
        self.dropna_feature_cols = parse_column_spec(dropna_feature_cols)

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        date_col = str(_resolve_date_column(df_raw, self.custom_date_col))
        if self.target not in df_raw.columns:
            raise ValueError(f"Target column '{self.target}' not found. Available columns: {list(df_raw.columns)}")

        if self.numeric_feature_cols:
            _validate_requested_columns(df_raw, self.numeric_feature_cols, 'numeric_feature')
        if self.prompt_context_cols:
            _validate_requested_columns(df_raw, self.prompt_context_cols, 'prompt_context')
        if self.dropna_feature_cols:
            _validate_requested_columns(df_raw, self.dropna_feature_cols, 'dropna_feature')

        if self.dropna_feature_cols:
            df_raw = df_raw.dropna(subset=self.dropna_feature_cols).reset_index(drop=True)

        if self.numeric_feature_cols:
            cols = [col for col in self.numeric_feature_cols if col not in [self.target, date_col]]
        else:
            cols = [col for col in df_raw.columns if col not in [self.target, date_col]]
        if not cols and self.features != 'S':
            raise ValueError('Custom dataset must contain at least one feature column besides date and target.')

        raw_prompt_df = df_raw[[date_col] + [col for col in self.prompt_context_cols if col in df_raw.columns] + [self.target]].copy()
        df_raw = df_raw[[date_col] + cols + [self.target]].copy()
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors='raise')
        df_raw = df_raw.set_index(date_col).sort_index().reset_index().rename(columns={'index': date_col})
        if not df_raw[date_col].is_monotonic_increasing:
            raise ValueError('Custom dataset timestamps must be monotonic increasing after sorting.')
        if df_raw[date_col].duplicated().any():
            raise ValueError('Custom dataset timestamps must be unique for a single-series forecasting setup.')
        raw_prompt_df[date_col] = pd.to_datetime(raw_prompt_df[date_col], errors='raise')
        raw_prompt_df = raw_prompt_df.set_index(date_col).sort_index().reset_index().rename(columns={'index': date_col})

        if self.train_end_date or self.val_end_date:
            if not (self.train_end_date and self.val_end_date):
                raise ValueError('Set both --train_end_date and --val_end_date for date-based custom splits.')
            border1s, border2s = _compute_date_boundaries(
                df_raw,
                date_col,
                self.seq_len,
                self.train_end_date,
                self.val_end_date,
            )
        else:
            border1s, border2s = _compute_ratio_boundaries(
                len(df_raw), self.seq_len, self.train_split_ratio, self.val_split_ratio, self.test_split_ratio, 'Custom ratio split'
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
            train_values = np.asarray(train_data, dtype=np.float32)
            df_values = np.asarray(df_data, dtype=np.float32)
            self.scaler.fit(train_values)
            data = np.asarray(self.scaler.transform(df_values), dtype=np.float32)
        else:
            data = np.asarray(df_data, dtype=np.float32)

        df_stamp = df_raw[[date_col]].iloc[border1:border2].copy()
        data_stamp = _build_time_features(df_stamp, date_col, self.timeenc, self.freq)

        self.data_x = np.asarray(data[border1:border2], dtype=np.float32)
        self.data_y = np.asarray(data[border1:border2], dtype=np.float32)
        self.data_stamp = data_stamp
        self.date_col = date_col
        self.prompt_df = raw_prompt_df.iloc[border1:border2].reset_index(drop=True)
        self.numeric_cols = cols
        self.model_feature_cols = cols + [self.target]
        self.target_channel_index = len(self.model_feature_cols) - 1
        if self.model_feature_cols[self.target_channel_index] != self.target:
            raise ValueError('Target column must be the last channel in Dataset_Custom model_feature_cols.')

    def _build_prompt_context(self, s_begin, s_end):
        if self.prompt_df.empty:
            return ''
        window_df = self.prompt_df.iloc[s_begin:s_end].copy()
        if window_df.empty:
            return ''
        if {'交通状况', '环境描述'}.intersection(window_df.columns) or {'holiday_星期', 'weather_天气'}.intersection(window_df.columns):
            return _build_park_prompt(window_df, self.target)
        return _build_generic_prompt(window_df, self.target, self.prompt_context_cols)

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
        prompt_context = self._build_prompt_context(s_begin, s_end)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, prompt_context

    def __len__(self):
        base_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        return base_len * self.enc_in if self.channel_independence else base_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len, self.label_len, self.pred_len = _resolve_window_sizes(size)

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return data

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts_last_window):] = ts_last_window
            insample_mask[i, -len(ts_last_window):] = 1.0
        return insample, insample_mask


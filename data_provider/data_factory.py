from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import random
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
}


def _build_loader_generator(args, flag):
    generator = torch.Generator()
    base_seed = int(getattr(args, 'seed', 2021))
    offset_map = {'train': 0, 'val': 1, 'test': 2}
    generator.manual_seed(base_seed + offset_map.get(flag, 0))
    return generator


def _seed_worker_global(worker_id, base_seed, flag_offset):
    worker_seed = int(base_seed) + int(flag_offset) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def data_provider(args, flag):
    Data = data_dict.get(args.data, Dataset_Custom)
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.eval_batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.eval_batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns,
            train_split_ratio=args.train_split_ratio,
            val_split_ratio=args.val_split_ratio,
            test_split_ratio=args.test_split_ratio,
            train_end_date=getattr(args, 'train_end_date', ''),
            val_end_date=getattr(args, 'val_end_date', ''),
            custom_date_col=args.custom_date_col,
            channel_independence=args.channel_independence,
            numeric_feature_cols=getattr(args, 'numeric_feature_cols', ''),
            prompt_context_cols=getattr(args, 'prompt_context_cols', ''),
            dropna_feature_cols=getattr(args, 'dropna_feature_cols', ''),
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        worker_init_fn=partial(
            _seed_worker_global,
            base_seed=int(getattr(args, 'seed', 2021)),
            flag_offset={'train': 0, 'val': 1000, 'test': 2000}.get(flag, 0),
        ),
        generator=_build_loader_generator(args, flag),
    )
    return data_set, data_loader

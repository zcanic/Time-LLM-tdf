from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import random
import torch

from data_provider_pretrain.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
from data_provider.data_loader import Dataset_Custom

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
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


def data_provider(args, data, data_path, pretrain=True, flag='train'):
    Data = data_dict.get(data, Dataset_Custom)
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

    common_kwargs = dict(
        root_path=args.root_path,
        data_path=data_path,
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
        pretrain_train_split_ratio=getattr(args, 'pretrain_train_split_ratio', 0.8),
        pretrain_eval_split_ratio=getattr(args, 'pretrain_eval_split_ratio', 0.2),
    )
    if Data in data_dict.values():
        data_set = Data(pretrain=pretrain, **common_kwargs)
    else:
        data_set = Data(**common_kwargs)
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

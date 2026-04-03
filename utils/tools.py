import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs

from tqdm import tqdm

plt.switch_backend('agg')


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    lr_adjust = {}
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def resolve_repo_path(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def build_accelerator(args):
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(getattr(args, 'find_unused_parameters', True))
    )
    use_deepspeed = bool(getattr(args, 'use_deepspeed', False))
    deepspeed_config = getattr(args, 'deepspeed_config', '')
    deepspeed_plugin = None
    if use_deepspeed:
        config_path = resolve_repo_path(deepspeed_config)
        if config_path is None or not config_path.exists():
            raise FileNotFoundError(
                f"DeepSpeed config not found: {deepspeed_config}. "
                "Set --use_deepspeed 0 for a local smoke test or pass a valid --deepspeed_config path."
            )
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=str(config_path))
    return Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


def cleanup_experiment_path(path_value, enabled=False, allowed_root=None):
    if not enabled:
        return False
    path = resolve_repo_path(path_value)
    if path is None or not path.exists():
        return False
    if allowed_root is not None:
        root_path = resolve_repo_path(allowed_root)
        if root_path is None:
            raise ValueError('Cleanup root path is invalid.')
        if path == root_path or root_path not in path.parents:
            raise ValueError(
                f'Refusing to delete {path} because it is not a child run directory under {root_path}.'
            )
    shutil.rmtree(path)
    return True


def load_content(args):
    if not bool(getattr(args, 'prompt_domain', False)):
        return ''

    prompt_text = getattr(args, 'prompt_text', '')
    if prompt_text:
        return prompt_text

    prompt_path = getattr(args, 'prompt_path', '')
    if prompt_path:
        resolved_prompt_path = resolve_repo_path(prompt_path)
        if resolved_prompt_path is None or not resolved_prompt_path.exists():
            raise FileNotFoundError(f'Prompt file not found: {prompt_path}')
        return resolved_prompt_path.read_text(encoding='utf-8')

    prompt_bank_name = getattr(args, 'prompt_bank_name', '')
    if prompt_bank_name:
        file = 'ETT' if 'ETT' in prompt_bank_name else prompt_bank_name
        prompt_bank_dir = getattr(args, 'prompt_bank_dir', '') or str(PROJECT_ROOT / 'dataset' / 'prompt_bank')
        resolved_prompt_bank_dir = resolve_repo_path(prompt_bank_dir)
        if resolved_prompt_bank_dir is None:
            raise FileNotFoundError(f'Prompt bank directory could not be resolved: {prompt_bank_dir}')
        prompt_file = resolved_prompt_bank_dir / f'{file}.txt'
        if not prompt_file.exists():
            raise FileNotFoundError(
                f'Prompt bank file not found: {prompt_file}. '
                'Pass --prompt_path, --prompt_text, or --prompt_bank_dir instead.'
            )
        return prompt_file.read_text(encoding='utf-8')

    return ''


def infer_data_dims(args, dataset):
    dataset_dim = getattr(dataset, 'enc_in', None)
    if dataset_dim is None:
        return

    if bool(getattr(args, 'infer_dims', True)) or args.enc_in <= 0:
        args.enc_in = int(dataset_dim)
    if bool(getattr(args, 'infer_dims', True)) or args.dec_in <= 0:
        args.dec_in = int(dataset_dim)
    if bool(getattr(args, 'infer_dims', True)) or args.c_out <= 0:
        args.c_out = int(dataset_dim)

    target_channel_index = getattr(dataset, 'target_channel_index', None)
    if target_channel_index is not None:
        args.target_channel_index = int(target_channel_index)


def default_dataset_description(args):
    dataset_name = getattr(args, 'data', 'dataset')
    target_name = getattr(args, 'target', 'target variable')
    freq = getattr(args, 'freq', 'unknown frequency')
    return (
        f'This dataset is a custom time series dataset named {dataset_name}. '
        f'The forecasting target is {target_name}. '
        f'The sampling frequency is {freq}. '
        'Use only the provided history and metadata instead of assuming an ETT electricity domain.'
    )


def parse_column_spec(column_spec):
    if column_spec is None:
        return []
    if isinstance(column_spec, (list, tuple)):
        return [str(col).strip() for col in column_spec if str(col).strip()]
    return [part.strip() for part in str(column_spec).split(',') if part.strip()]


def apply_dataset_profile(args):
    profile = getattr(args, 'dataset_profile', '')
    data_path = str(getattr(args, 'data_path', ''))
    is_park_profile = profile == 'park_featured' or data_path.endswith('park_featured_data.csv')
    if not is_park_profile:
        return

    args.data = 'park_featured'
    if str(getattr(args, 'data_path', '')) in {'', 'ETTh1.csv'}:
        args.data_path = 'park_featured_data.csv'
    if str(getattr(args, 'root_path', '')).endswith('dataset'):
        args.root_path = str(PROJECT_ROOT / 'data_process_and_data_to_use')
    args.custom_date_col = args.custom_date_col or '时间戳'
    args.target = 'number'
    args.features = 'MS'
    args.freq = '15min'
    args.prompt_domain = 1
    args.model = 'TimeLLM'
    args.llm_model = 'BERT'
    args.llm_dim = 768
    args.llm_model_path = 'hfl/chinese-roberta-wwm-ext'
    args.tokenizer_path = 'hfl/chinese-roberta-wwm-ext'

    if not getattr(args, 'numeric_feature_cols', ''):
        args.numeric_feature_cols = ','.join([
            'feat_baidu_lag1d',
            'feat_baidu_diff_1d',
            'feat_baidu_ma_3d',
            'feat_baidu_ma_7d',
            'feat_baidu_ma_spread_3d_7d',
        ])
    if not getattr(args, 'dropna_feature_cols', ''):
        args.dropna_feature_cols = args.numeric_feature_cols
    if not getattr(args, 'prompt_context_cols', ''):
        args.prompt_context_cols = ','.join([
            '交通状况',
            '环境描述',
            'weather_天气',
            'weather_最低气温_摄氏度',
            'weather_平均气温_摄氏度',
            'weather_最高气温_摄氏度',
            'weather_总降水量_毫米',
            'holiday_星期',
            'holiday_是否周末',
            'holiday_是否节假日放假',
            'holiday_节假日名称',
            'holiday_是否调休上班',
            'holiday_日期标签',
        ])
    if not getattr(args, 'dataset_description', ''):
        args.dataset_description = (
            'This dataset records timestamp-ordered Tiantan park visitor-count observations on a 15-minute intra-day grid with overnight closure gaps. '
            'The target is the next observed-row number value. Prompt context may include observed traffic status, environment description, '
            'weather conditions, weekday and holiday information from the observed window only. '
            'Baidu-derived lag features are numeric assistance covariates and rows with missing required Baidu features are excluded.'
        )


def default_llm_source(llm_model):
    model_id_map = {
        'LLAMA': 'huggyllama/llama-7b',
        'GPT2': 'openai-community/gpt2',
        'BERT': 'google-bert/bert-base-uncased',
    }
    if llm_model not in model_id_map:
        raise ValueError(f'Unsupported llm_model: {llm_model}')
    return model_id_map[llm_model]


def unpack_model_batch(batch):
    if len(batch) == 4:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        return batch_x, batch_y, batch_x_mark, batch_y_mark, None
    if len(batch) == 5:
        batch_x, batch_y, batch_x_mark, batch_y_mark, prompt_context = batch
        return batch_x, batch_y, batch_x_mark, batch_y_mark, prompt_context
    raise ValueError(f'Unexpected batch structure length: {len(batch)}')


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(vali_loader)):
            batch_x, batch_y, batch_x_mark, batch_y_mark, prompt_context = unpack_model_batch(batch)
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_context=prompt_context)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_context=prompt_context)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_context=prompt_context)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_context=prompt_context)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = int(getattr(args, 'target_channel_index', args.enc_in - 1)) if args.features == 'MS' else 0
            if outputs.shape[-1] <= f_dim or batch_y.shape[-1] <= f_dim:
                raise ValueError(
                    f'Target channel index {f_dim} is out of bounds for outputs {tuple(outputs.shape)} and batch_y {tuple(batch_y.shape)}.'
                )
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)

            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss



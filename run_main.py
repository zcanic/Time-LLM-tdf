import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    build_accelerator,
    cleanup_experiment_path,
    default_dataset_description,
    infer_data_dims,
    load_content,
    apply_dataset_profile,
    resolve_repo_path,
    default_llm_source,
    unpack_model_batch,
    vali,
)

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear, TimeLLM]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTm1', help='dataset type')
parser.add_argument('--dataset_profile', type=str, default='', help='optional dataset profile, e.g. park_featured')
parser.add_argument('--root_path', type=str, default=str(resolve_repo_path('dataset')), help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default=str(resolve_repo_path('checkpoints')), help='location of model checkpoints')
parser.add_argument('--train_split_ratio', type=float, default=0.7, help='train split ratio for custom datasets')
parser.add_argument('--val_split_ratio', type=float, default=0.1, help='validation split ratio for custom datasets')
parser.add_argument('--test_split_ratio', type=float, default=0.2, help='test split ratio for custom datasets')
parser.add_argument('--train_end_date', type=str, default='', help='optional inclusive train end date for custom datasets')
parser.add_argument('--val_end_date', type=str, default='', help='optional inclusive validation end date for custom datasets')
parser.add_argument('--custom_date_col', type=str, default='', help='optional timestamp column name for custom datasets')
parser.add_argument('--channel_independence', type=int, default=0, help='1: one channel per sample, 0: keep all channels together')
parser.add_argument('--numeric_feature_cols', type=str, default='', help='comma-separated numeric covariate columns to keep in the encoder input')
parser.add_argument('--prompt_context_cols', type=str, default='', help='comma-separated observed-window columns to summarize into prompt context')
parser.add_argument('--dropna_feature_cols', type=str, default='', help='comma-separated columns that must be non-null before a row is eligible for training')

# forecasting task
parser.add_argument('--seq_len', type=int, default=-1, help='input sequence length; set explicitly for reproducible runs')
parser.add_argument('--label_len', type=int, default=-1, help='start token length; set explicitly for reproducible runs')
parser.add_argument('--pred_len', type=int, default=-1, help='prediction sequence length; set explicitly for reproducible runs')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=-1, help='encoder input size; <=0 infers from dataset')
parser.add_argument('--dec_in', type=int, default=-1, help='decoder input size; <=0 infers from dataset')
parser.add_argument('--c_out', type=int, default=-1, help='output size; <=0 infers from dataset')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--llm_model_path', type=str, default='', help='optional HuggingFace model path or local directory override')
parser.add_argument('--tokenizer_path', type=str, default='', help='optional tokenizer path override')
parser.add_argument('--local_files_only', type=int, default=0, help='force local model/tokenizer loading only')
parser.add_argument('--dataset_description', type=str, default='', help='dataset description used when prompt files are not provided')
parser.add_argument('--prompt_text', type=str, default='', help='inline prompt text override')
parser.add_argument('--prompt_path', type=str, default='', help='optional prompt file path')
parser.add_argument('--prompt_bank_name', type=str, default='', help='optional prompt bank file stem to load')
parser.add_argument('--prompt_bank_dir', type=str, default='', help='optional directory containing prompt bank files')
parser.add_argument('--top_k', type=int, default=5, help='number of lag candidates injected into the prompt')
parser.add_argument('--num_tokens', type=int, default=1000, help='projected vocabulary token count used by the reprogramming layer')
parser.add_argument('--prompt_max_length', type=int, default=2048, help='maximum tokenizer prompt length')
parser.add_argument('--patch_embedding_dtype', type=str, default='auto', choices=['auto', 'float32', 'float16', 'bfloat16'], help='dtype used before patch embedding')


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--infer_dims', type=int, default=1, help='infer enc_in/dec_in/c_out from the dataset')
parser.add_argument('--use_deepspeed', type=int, default=0, help='enable DeepSpeed-backed Accelerator setup')
parser.add_argument('--deepspeed_config', type=str, default='ds_config_zero2.json', help='DeepSpeed config path when enabled')
parser.add_argument('--find_unused_parameters', type=int, default=1, help='DDP find_unused_parameters flag')
parser.add_argument('--cleanup_checkpoints', type=int, default=0, help='delete only this run checkpoint directory after finishing')

args = parser.parse_args()
apply_dataset_profile(args)
seed = int(args.seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if hasattr(torch, 'use_deterministic_algorithms'):
    torch.use_deterministic_algorithms(True, warn_only=True)
args.root_path = str(resolve_repo_path(args.root_path))
args.checkpoints = str(resolve_repo_path(args.checkpoints))
if not args.llm_model_path:
    args.llm_model_path = default_llm_source(args.llm_model)
if not args.tokenizer_path:
    args.tokenizer_path = args.llm_model_path
if args.data != 'm4' and min(args.seq_len, args.label_len, args.pred_len) <= 0:
    raise ValueError('Set --seq_len, --label_len, and --pred_len explicitly.')
accelerator = build_accelerator(args)
path = ''

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    infer_data_dims(args, train_data)
    if not hasattr(args, 'target_channel_index'):
        args.target_channel_index = args.enc_in - 1
    args.content = load_content(args)
    if not args.dataset_description:
        args.dataset_description = default_dataset_description(args)

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, batch in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x, batch_y, batch_x_mark, batch_y_mark, prompt_context = unpack_model_batch(batch)

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_context=prompt_context)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_context=prompt_context)

                    f_dim = int(getattr(args, 'target_channel_index', args.enc_in - 1)) if args.features == 'MS' else 0
                    if outputs.shape[-1] <= f_dim or batch_y.shape[-1] <= f_dim:
                        raise ValueError(
                            f'Target channel index {f_dim} is out of bounds for outputs {tuple(outputs.shape)} and batch_y {tuple(batch_y.shape)}.'
                        )
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_context=prompt_context)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_context=prompt_context)

                f_dim = int(getattr(args, 'target_channel_index', args.enc_in - 1)) if args.features == 'MS' else 0
                if outputs.shape[-1] <= f_dim or batch_y.shape[-1] <= f_dim:
                    raise ValueError(
                        f'Target channel index {f_dim} is out of bounds for outputs {tuple(outputs.shape)} and batch_y {tuple(batch_y.shape)}.'
                    )
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    cleaned = cleanup_experiment_path(path, bool(args.cleanup_checkpoints), args.checkpoints)
    if cleaned:
        accelerator.print(f'success delete checkpoints: {path}')

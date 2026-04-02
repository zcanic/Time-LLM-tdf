model_name=${MODEL_NAME:-TimeLLM}
train_epochs=${TRAIN_EPOCHS:-20}
learning_rate=${LEARNING_RATE:-0.001}
llm_layers=${LLM_LAYERS:-6}
main_process_port=${MAIN_PROCESS_PORT:-29517}
num_processes=${NUM_PROCESSES:-1}
mixed_precision=${MIXED_PRECISION:-no}
batch_size=${BATCH_SIZE:-24}
d_model=${D_MODEL:-32}
d_ff=${D_FF:-128}
root_path=${ROOT_PATH:-./dataset/ETT-small/}
data_path=${DATA_PATH:-ETTh1.csv}
enc_in=${ENC_IN:--1}
dec_in=${DEC_IN:--1}
c_out=${C_OUT:--1}

comment=${MODEL_COMMENT:-TimeLLM-ETTh1}
launch_mode=""
if [ "$num_processes" -gt 1 ]; then
  launch_mode="--multi_gpu --num_processes $num_processes --main_process_port $main_process_port"
fi

run_case() {
  pred_len=$1
  lr=$2
  lradj=$3
  accelerate launch $launch_mode --mixed_precision $mixed_precision run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "ETTh1_512_${pred_len}" \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len $pred_len \
    --factor 3 \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --des Exp \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $lr \
    --llm_layers $llm_layers \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --use_deepspeed 0 \
    --cleanup_checkpoints 0 \
    --infer_dims 1 \
    ${lradj:+--lradj $lradj}
}

run_case 96 "$learning_rate" ""
run_case 192 0.02 ""
run_case 336 0.001 COS
run_case 720 "$learning_rate" ""

#!/bin/bash
#
# Zhenhao Ge, 2024-04-23

ROOT_DIR=/home/users/zge/code/repo/mqtts

CURRENT_DIR=$PWD
[[ $CURRENT_DIR != $ROOT_DIR ]] && cd $ROOT_DIR \
  && echo "change current dir to: $ROOT_DIR"

model_path=${ROOT_DIR}/ckpt/OTS/transformer.ckpt
config_path=${ROOT_DIR}/ckpt/OTS/config.json 
batch_size=1
num_samples=24
noise_seed=1
CUDA_DEVICE=0

OUTPUT_DIR=$ROOT_DIR/outputs/infer_samples_${noise_seed}
mkdir -p $OUTPUT_DIR

echo "model path: ${model_path}"
echo "config path: ${config_path}"
echo "batch size: ${batch_size}"
echo "num of samples: ${num_samples}"
echo "noise seed: ${noise_seed}"
echo "GPU device: ${CUDA_DEVICE}"
echo "output dir: ${OUTPUT_DIR}"

CUDA_VISIBLE_DIVICES=$CUDA_DIVICE python infer.py \
  --phonemizer_dict_path en_us_cmudict_forward.pt \
  --model_path  $model_path \
  --config_path $config_path \
  --input_path speaker_to_text.json \
  --outputdir $OUTPUT_DIR \
  --batch_size $batch_size \
  --num_samples $num_samples \
  --top_p 0.8 \
  --min_top_k 2 \
  --max_output_length 100000 \
  --phone_context_window 3 \
  --noise_seed $noise_seed \
  --device 0 \
  --clean_speech_prior

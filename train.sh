export DETECTRON2_DISABLE_LOGGING=1
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python main.py \
        --config-file configs/vitmatte_s_am2k_100ep.py \
        --num-gpus 1 \
        --num-machines 1 \
        --dist-url "tcp://127.0.0.1:1416"


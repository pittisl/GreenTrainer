MODEL_NAME="facebook/opt-2.7b"


# GreenTrainer-0.5
echo '### GreenTrainer-0.5'
python3 main.py --model_name $MODEL_NAME \
                --dataset_name piqa \
                --scheme green_trainer \
                --max_input_length 400 \
                --max_output_length 4 \
                --batch_size 4 \
                --rho 0.5 \
                --task qa

# LoRA
echo '### LoRA'
python3 main.py --model_name $MODEL_NAME \
                --dataset_name piqa \
                --scheme baselines \
                --train_type lora \
                --max_input_length 400 \
                --max_output_length 4 \
                --batch_size 4 \
                --task qa \
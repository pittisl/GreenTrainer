MODEL_NAME="facebook/opt-2.7b"


# GreenTrainer-0.5
echo '### GreenTrainer-0.5'
python3 main.py --model_name $MODEL_NAME \
                --dataset_name webquestions \
                --scheme green_trainer \
                --max_input_length 160 \
                --max_output_length 64 \
                --batch_size 4 \
                --rho 0.5 \
                --task qa \

# LoRA
echo '### LoRA'
python3 main.py --model_name $MODEL_NAME \
                --dataset_name webquestions \
                --scheme baselines \
                --train_type lora \
                --max_input_length 160 \
                --max_output_length 64 \
                --batch_size 4 \
                --task qa \
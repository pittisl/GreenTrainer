import os
from train import Trainer, Green_Trainer
from models import load_text_generation_model
from data import dataset_loader
from utils import make_folders

# TODO Evaluation Plan
# (1) text summarization: 
# facebook/opt-2.7b
# facebook/opt-6.7b
# bloomz-3b
# bloomz-7b
# decapoda-research/llama-7b-hf
# scitldr -> [bs=16, in=512, out=256], 
# samsum -> [bs=16, in=512, out=256], 
# xsum, cnndailymail -> [bs=8, in=1024, out=256]
# (2) abstractive question answering:
# durocp, durocs -> [bs=8, in=1024, out=128], 
# (3) impact of model complexity:
# facebook/opt-350m, facebook/opt-1.3b, facebook/opt-2.7b, facebook/opt-6.7b, facebook/opt-13b (13B)
# scitldr -> [bs=16, in=512, out=256],

make_folders("logs", "saved_models")

os.environ["TOKENIZERS_PARALLELISM"] = "true"

scheme = 'baselines' # baselines or green_trainer
train_type = "full_finetuning"

task = 'train' # train or evaluate

model_name = "facebook/opt-125m" # "facebook/opt-125m" "gpt2"
dataset_name = "samsum"
max_input_length = 512
max_output_length = 512
batch_size = 4

if scheme == 'baselines':

    train_type = "full_finetuning" # "full_finetuning"
    model_path = f"saved_models/{model_name.replace('/', '_')}_{train_type}"
    
    model = load_text_generation_model(
        model_name, train_type,
        output_attentions=False,
    )
    train_loader, tokenizer = dataset_loader(
        dataset_name=dataset_name,
        split="train",
        tokenizer_name=model_name,
        model_name=model_name,
        max_input_length=max_input_length, 
        batch_size=batch_size,
        shuffle=True,
        keep_in_memory=True,
        print_info=False,
    )

    val_loader, _ = dataset_loader(
        dataset_name=dataset_name,
        split="validation",
        tokenizer_name=model_name,
        model_name=model_name,
        max_input_length=max_input_length, 
        batch_size=batch_size,
        shuffle=False,
        keep_in_memory=True,
        print_info=False,
    )

    test_loader, _ = dataset_loader(
        dataset_name=dataset_name,
        split="test",
        tokenizer_name=model_name,
        model_name=model_name,
        max_input_length=max_input_length, 
        batch_size=batch_size,
        shuffle=False,
        keep_in_memory=True,
        print_info=False,
    )

    my_trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        train_type=train_type,
        tokenizer=tokenizer,
        max_output_length=max_output_length,
        model_path=model_path,
    )
    if task == 'train':
        my_trainer.train(
            learning_rate=2e-5,
            num_epochs=5,
            log_dir=f"logs/{model_name.replace('/', '_')}_{train_type}"
        )
    elif task == 'evaluate':
        my_trainer.evaluate()

elif scheme == 'green_trainer':
    
    train_type = scheme
    rho = 0.7
    model_path = f"saved_models/{model_name.replace('/', '_')}_{train_type}_{rho}"

    model = load_text_generation_model(
        model_name, train_type,
        output_attentions=False,
    )
    train_loader, tokenizer = dataset_loader(
        dataset_name=dataset_name,
        split="train",
        tokenizer_name=model_name,
        model_name=model_name,
        max_input_length=max_input_length, 
        batch_size=batch_size,
        shuffle=True,
        keep_in_memory=True,
        print_info=False,
    )

    val_loader, _ = dataset_loader(
        dataset_name=dataset_name,
        split="validation",
        tokenizer_name=model_name,
        model_name=model_name,
        max_input_length=max_input_length, 
        batch_size=batch_size,
        shuffle=False,
        keep_in_memory=True,
        print_info=False,
    )

    test_loader, _ = dataset_loader(
        dataset_name=dataset_name,
        split="test",
        tokenizer_name=model_name,
        model_name=model_name,
        max_input_length=max_input_length, 
        batch_size=batch_size,
        shuffle=False,
        keep_in_memory=True,
        print_info=False,
    )

    my_trainer = Green_Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        model_type=model_name,
        train_type=train_type,
        tokenizer=tokenizer,
        max_output_length=max_output_length,
        model_path=model_path,
    )
    if task == 'train':
        my_trainer.train(
            learning_rate=2e-5,
            num_epochs=5,
            input_length=max_input_length,
            output_length=max_output_length,
            rho=rho,
            log_dir=f"logs/{model_name.replace('/', '_')}_{train_type}_{rho}"
        )
    elif task == 'evaluate':
        my_trainer.evaluate()


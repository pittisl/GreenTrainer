import os
from train import Trainer, Green_Trainer
from models import load_text_generation_model
from data import dataset_loader
from utils import make_folders

# Evaluation Plan
# (1) text summarization: 
# google/flan-t5-base (250M) (prefix="summarize: ")
# samsum -> [bs=16, in=512, out=256], 
# scitldr -> [bs=16, in=512, out=256], 
# xsum, cnndailymail -> [bs=8, in=1024, out=256]
# (2) abstractive question answering:
# google/flan-t5-large (780M), google/flan-t5-xl (3B)
# durocp, durocs -> [bs=8, in=1024, out=128], 
# (3) impact of model complexity:
# google/flan-t5-small (80M), google/flan-t5-base (250M), google/flan-t5-large (780M), google/flan-t5-xl (3B) (prefix="summarize: ")
# scitldr -> [bs=16, in=512, out=256],

make_folders("logs", "saved_models")

os.environ["TOKENIZERS_PARALLELISM"] = "true"

scheme = 'baselines' # baselines or green_trainer
train_type = 'full_finetuning' # full_finetuning, lora, prefix_tuning, adalora

task = 'train' # train or evaluate

model_name = "google/flan-t5-base"
dataset_name = "duorcs"
prefix = ""
max_input_length = 1024
max_output_length = 256
batch_size = 16

if scheme == 'baselines':
    
    model_path = f"saved_models/{model_name.replace('/', '_')}_{train_type}"
    
    model = load_text_generation_model(
        model_name, train_type,
        output_attentions=False,
    )
    train_loader, tokenizer = dataset_loader(
        dataset_name=dataset_name,
        split="train",
        tokenizer_name=model_name,
        max_input_length=max_input_length, 
        max_output_length=max_output_length,
        batch_size=batch_size,
        prefix=prefix,
        shuffle=True,
        keep_in_memory=True,
        print_info=False,
    )

    val_loader, _ = dataset_loader(
        dataset_name=dataset_name,
        split="validation",
        tokenizer_name=model_name,
        max_input_length=max_input_length, 
        max_output_length=max_output_length,
        batch_size=batch_size,
        prefix=prefix,
        shuffle=False,
        keep_in_memory=True,
        print_info=False,
    )

    test_loader, _ = dataset_loader(
        dataset_name=dataset_name,
        split="test",
        tokenizer_name=model_name,
        max_input_length=max_input_length, 
        max_output_length=max_output_length,
        batch_size=batch_size,
        prefix=prefix,
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
    rho = 0.5
    model_path = f"saved_models/{model_name.replace('/', '_')}_{train_type}_{rho}"

    model = load_text_generation_model(
        model_name, train_type,
        output_attentions=False,
    )
    train_loader, tokenizer = dataset_loader(
        dataset_name=dataset_name,
        split="train",
        tokenizer_name=model_name,
        max_input_length=max_input_length, 
        max_output_length=max_output_length,
        batch_size=batch_size,
        prefix=prefix,
        shuffle=True,
        keep_in_memory=True,
        print_info=False,
    )

    val_loader, _ = dataset_loader(
        dataset_name=dataset_name,
        split="validation",
        tokenizer_name=model_name,
        max_input_length=max_input_length, 
        max_output_length=max_output_length,
        batch_size=batch_size,
        prefix=prefix,
        shuffle=False,
        keep_in_memory=True,
        print_info=False,
    )

    test_loader, _ = dataset_loader(
        dataset_name=dataset_name,
        split="test",
        tokenizer_name=model_name,
        max_input_length=max_input_length, 
        max_output_length=max_output_length,
        batch_size=batch_size,
        prefix=prefix,
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


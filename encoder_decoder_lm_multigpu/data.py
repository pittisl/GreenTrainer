from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformers import default_data_collator
from torch.utils.data import DataLoader


def dataset_loader(
    dataset_name,
    split,
    tokenizer_name,
    max_input_length,
    max_output_length,
    batch_size,
    prefix="",
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load a dataset ready for training
    
    Supported: `xsum`, `samsum`, `cnn_dailymail`, 
    `scitldr`, `duorcp`, `duorcs`.
    """
    
    if dataset_name == "xsum":
        dataloader, tokenizer = load_xsum(
            split=split,
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            batch_size=batch_size,
            prefix=prefix,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "samsum":
        dataloader, tokenizer = load_samsum(
            split=split,
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            batch_size=batch_size,
            prefix=prefix,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "dialogsum":
        dataloader, tokenizer = load_dialogsum(
            split=split,
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            batch_size=batch_size,
            prefix=prefix,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "cnn_dailymail":
        dataloader, tokenizer = load_cnndailymail(
            split=split,
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            batch_size=batch_size,
            prefix=prefix,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "scitldr":
        dataloader, tokenizer = load_scitldr(
            split=split,
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            batch_size=batch_size,
            prefix=prefix,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "duorcs":
        dataloader, tokenizer = load_duorcs(
            split=split,
            tokenizer_name=tokenizer_name,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            batch_size=batch_size,
            prefix=prefix,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    else:
        raise NotImplementedError(f"{dataset_name} hasn't been implemented yet!")
    
    return dataloader, tokenizer


def load_xsum(
    split,
    tokenizer_name,
    max_input_length,
    max_output_length,
    batch_size,
    prefix="",
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load xsum dataset
    train: 204k, valid: 11.3k, test: 11.3k
    """
    
    dataset = load_dataset(
        "xsum", split=split, 
        keep_in_memory=keep_in_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 204k, valid: 11.3k, test: 11.3k")
    
    def preprocess_fn(examples):
        prefixed_inputs = [
            prefix + doc for doc in examples["document"]
        ]      
        tokenized_inputs = tokenizer(
            prefixed_inputs, 
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )    
        tokenized_labels = tokenizer(
            examples["summary"], 
            max_length=max_output_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
        )
        labels = tokenized_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    dataloader = DataLoader(
        processed_dataset, shuffle=shuffle, 
        collate_fn=default_data_collator, 
        batch_size=batch_size, pin_memory=True,
    )
    return dataloader, tokenizer


def load_samsum(
    split,
    tokenizer_name,
    max_input_length,
    max_output_length,
    batch_size,
    prefix="",
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load samsum dataset
    train: 14.7k, valid: 819, test: 818
    """
    
    dataset = load_dataset(
        "samsum", split=split, 
        keep_in_memory=keep_in_memory
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 14.7k, valid: 819, test: 818")
    
    def preprocess_fn(examples):
        prefixed_inputs = [
            prefix + doc for doc in examples["dialogue"]
        ]      
        tokenized_inputs = tokenizer(
            prefixed_inputs, 
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )    
        tokenized_labels = tokenizer(
            examples["summary"], 
            max_length=max_output_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
        )
        labels = tokenized_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    dataloader = DataLoader(
        processed_dataset, shuffle=shuffle, 
        collate_fn=default_data_collator, 
        batch_size=batch_size, pin_memory=True,
    )
    return dataloader, tokenizer


def load_dialogsum(
    split,
    tokenizer_name,
    max_input_length,
    max_output_length,
    batch_size,
    prefix="",
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load dialogsum dataset
    train: 12.5k, valid: 500, test: 1.5k
    """
    
    dataset = load_dataset(
        "knkarthick/dialogsum", split=split, 
        keep_in_memory=keep_in_memory
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 12.5k, valid: 500, test: 1.5k")
    
    def preprocess_fn(examples):
        prefixed_inputs = [
            prefix + doc for doc in examples["dialogue"]
        ]      
        tokenized_inputs = tokenizer(
            prefixed_inputs, 
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )    
        tokenized_labels = tokenizer(
            examples["summary"], 
            max_length=max_output_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
        )
        labels = tokenized_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    dataloader = DataLoader(
        processed_dataset, shuffle=shuffle, 
        collate_fn=default_data_collator, 
        batch_size=batch_size, pin_memory=True,
    )
    return dataloader, tokenizer


def load_cnndailymail(
    split,
    tokenizer_name,
    max_input_length,
    max_output_length,
    batch_size,
    prefix="",
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load cnn dailymail dataset
    train: 287k, valid: 13.4k, test: 11.5k
    """
    
    dataset = load_dataset(
        "cnn_dailymail", "3.0.0", split=split, 
        keep_in_memory=keep_in_memory
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 287k, valid: 13.4k, test: 11.5k")
    
    def preprocess_fn(examples):
        prefixed_inputs = [
            prefix + doc for doc in examples["article"]
        ]      
        tokenized_inputs = tokenizer(
            prefixed_inputs, 
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )    
        tokenized_labels = tokenizer(
            examples["highlights"], 
            max_length=max_output_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
        )
        labels = tokenized_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    dataloader = DataLoader(
        processed_dataset, shuffle=shuffle, 
        collate_fn=default_data_collator, 
        batch_size=batch_size, pin_memory=True,
    )
    return dataloader, tokenizer


def load_scitldr(
    split,
    tokenizer_name,
    max_input_length,
    max_output_length,
    batch_size,
    prefix="",
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load scitldr/Abstract dataset
    train: 1.99k, valid: 619, test: 618
    """
    
    dataset = load_dataset(
        "allenai/scitldr", "Abstract", split=split, 
        keep_in_memory=keep_in_memory
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 1.99k, valid: 619, test: 618")
    
    def preprocess_fn(examples):
        prefixed_inputs = [
            prefix + "".join(doc) for doc in examples["source"]
        ]      
        tokenized_inputs = tokenizer(
            prefixed_inputs, 
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        target = [doc[0] for doc in examples["target"]]
        tokenized_labels = tokenizer(
            target, 
            max_length=max_output_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
        )
        labels = tokenized_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    dataloader = DataLoader(
        processed_dataset, shuffle=shuffle, 
        collate_fn=default_data_collator, 
        batch_size=batch_size, pin_memory=True,
    )
    return dataloader, tokenizer


def load_duorcs(
    split,
    tokenizer_name,
    max_input_length,
    max_output_length,
    batch_size,
    prefix="",
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load duorc/SelfRC dataset
    train: 60.7k, valid: 13k, test: 12.6k
    """
    
    def duorc_parser(example):
        return [example["plot"]], [example["question"]], [example["answers"][0] if len(example["answers"]) > 0 else ""]
    
    dataset = load_dataset(
        "duorc", "SelfRC", split=split, 
        keep_in_memory=keep_in_memory
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 60.7k, valid: 13k, test: 12.6k")
    
    questions_list = []
    answers_list = []
    contexts_list = []
    
    for row in dataset:
        contexts, questions, answers = duorc_parser(row)
        questions_list += contexts
        answers_list += questions
        contexts_list += answers
    
    dataset = Dataset.from_dict({"question": questions_list, "context": contexts_list, "answer": answers_list})
    
    def preprocess_fn(examples):
        prefixed_inputs = [
            f"QUESTION: {question} CONTEXT: {context}" for question, context in zip(examples['question'], examples['context'])
        ]      
        tokenized_inputs = tokenizer(
            prefixed_inputs, 
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )    
        tokenized_labels = tokenizer(
            examples["answer"], 
            max_length=max_output_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
        )
        labels = tokenized_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    dataloader = DataLoader(
        processed_dataset, shuffle=shuffle, 
        collate_fn=default_data_collator, 
        batch_size=batch_size, pin_memory=True,
    )
    return dataloader, tokenizer
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformers import default_data_collator
from torch.utils.data import DataLoader
import copy


def dataset_loader(
    dataset_name,
    split,
    tokenizer_name,
    model_name,
    max_input_length,
    batch_size,
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
            model_name=model_name,
            max_input_length=max_input_length,
            batch_size=batch_size,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "samsum":
        dataloader, tokenizer = load_samsum(
            split=split,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            max_input_length=max_input_length,
            batch_size=batch_size,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "dialogsum":
        dataloader, tokenizer = load_dialogsum(
            split=split,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            max_input_length=max_input_length,
            batch_size=batch_size,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "cnn_dailymail":
        dataloader, tokenizer = load_cnndailymail(
            split=split,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            max_input_length=max_input_length,
            batch_size=batch_size,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "scitldr":
        dataloader, tokenizer = load_scitldr(
            split=split,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            max_input_length=max_input_length,
            batch_size=batch_size,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "piqa":
        dataloader, tokenizer = load_piqa(
            split=split,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            max_input_length=max_input_length,
            batch_size=batch_size,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "webquestions":
        dataloader, tokenizer = load_webquestions(
            split=split,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            max_input_length=max_input_length,
            batch_size=batch_size,
            shuffle=shuffle,
            keep_in_memory=keep_in_memory,
            print_info=print_info,
        )
    elif dataset_name == "duorcs":
        dataloader, tokenizer = load_duorcs(
            split=split,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            max_input_length=max_input_length,
            batch_size=batch_size,
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
    model_name,
    max_input_length,
    batch_size,
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
    if 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 204k, valid: 11.3k, test: 11.3k")    
        
    def _tokenize_fn(strings):
        """Tokenize a list of strings, memorize source length"""
        tokenizer.padding_side = "right" 
        tokenized_list = [
            tokenizer(
                text,
                max_length=max_input_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            mask.sum().item() - 1 for mask in attention_mask
        ]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )   
    
    def _truncate_context(context, max_length=896):
        context_tokens = tokenizer(context)
        truncate_ratio = max_length / (len(context_tokens["input_ids"]) + 1e-5)
        if truncate_ratio < 1:
            return context[:int(len(context) * truncate_ratio)]
        else:
            return context 
    
    def preprocess_fn(raw_examples):
        """preprocess example strings, mask source part in labels"""
        sources = [
            f"{_truncate_context(context)} TL;DR: " for context in raw_examples['document']
        ]
        examples = [
            f"{_truncate_context(context)} TL;DR: {summary}" for context, summary in zip(raw_examples['document'], raw_examples['summary'])
        ]
        # left-padded source for validation & testing
        tokenizer.padding_side = "left"
        
        if 'gpt2' in model_name:
            tokenizer.pad_token = tokenizer.eos_token
            
        lp_sources = tokenizer(
            sources,
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100 # mask source
            label[label == tokenizer.pad_token_id] = -100 # mask paddings
        return dict(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            input_ids_lens=sources_tokenized["input_ids_lens"],
            lp_sources=lp_sources["input_ids"])
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "input_ids_lens", "lp_sources"]
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
    model_name,
    max_input_length,
    batch_size,
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
    if 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 14.7k, valid: 819, test: 818")    
        
    def _tokenize_fn(strings):
        """Tokenize a list of strings, memorize source length"""
        tokenizer.padding_side = "right" 
        tokenized_list = [
            tokenizer(
                text,
                max_length=max_input_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            mask.sum().item() - 1 for mask in attention_mask
        ]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )   
    
    def _truncate_context(context, max_length=416):
        context_tokens = tokenizer(context)
        truncate_ratio = max_length / (len(context_tokens["input_ids"]) + 1e-5)
        if truncate_ratio < 1:
            return context[:int(len(context) * truncate_ratio)]
        else:
            return context 
    
    def preprocess_fn(raw_examples):
        """preprocess example strings, mask source part in labels"""
        sources = [
            f"{_truncate_context(context)} TL;DR: " for context in raw_examples['dialogue']
        ]
        examples = [
            f"{_truncate_context(context)} TL;DR: {summary}" for context, summary in zip(raw_examples['dialogue'], raw_examples['summary'])
        ]
        # left-padded source for validation & testing
        tokenizer.padding_side = "left"
        
        if 'gpt2' in model_name:
            tokenizer.pad_token = tokenizer.eos_token
            
        lp_sources = tokenizer(
            sources,
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100 # mask source
            label[label == tokenizer.pad_token_id] = -100 # mask paddings
        return dict(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            input_ids_lens=sources_tokenized["input_ids_lens"],
            lp_sources=lp_sources["input_ids"])
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "input_ids_lens", "lp_sources"]
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
    model_name,
    max_input_length,
    batch_size,
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
    if 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 12.5k, valid: 500, test: 1.5k")    
        
    def _tokenize_fn(strings):
        """Tokenize a list of strings, memorize source length"""
        tokenizer.padding_side = "right" 
        tokenized_list = [
            tokenizer(
                text,
                max_length=max_input_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            mask.sum().item() - 1 for mask in attention_mask
        ]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )   
    
    def _truncate_context(context, max_length=416):
        context_tokens = tokenizer(context)
        truncate_ratio = max_length / (len(context_tokens["input_ids"]) + 1e-5)
        if truncate_ratio < 1:
            return context[:int(len(context) * truncate_ratio)]
        else:
            return context 
    
    def preprocess_fn(raw_examples):
        """preprocess example strings, mask source part in labels"""
        sources = [
            f"{_truncate_context(context)} TL;DR: " for context in raw_examples['dialogue']
        ]
        examples = [
            f"{_truncate_context(context)} TL;DR: {summary}" for context, summary in zip(raw_examples['dialogue'], raw_examples['summary'])
        ]
        # left-padded source for validation & testing
        tokenizer.padding_side = "left"
        
        if 'gpt2' in model_name:
            tokenizer.pad_token = tokenizer.eos_token
            
        lp_sources = tokenizer(
            sources,
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100 # mask source
            label[label == tokenizer.pad_token_id] = -100 # mask paddings
        return dict(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            input_ids_lens=sources_tokenized["input_ids_lens"],
            lp_sources=lp_sources["input_ids"])
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "input_ids_lens", "lp_sources"]
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
    model_name,
    max_input_length,
    batch_size,
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
    if 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 287k, valid: 13.4k, test: 11.5k")    
        
    def _tokenize_fn(strings):
        """Tokenize a list of strings, memorize source length"""
        tokenizer.padding_side = "right" 
        tokenized_list = [
            tokenizer(
                text,
                max_length=max_input_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            mask.sum().item() - 1 for mask in attention_mask
        ]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )   
    
    def _truncate_context(context, max_length=896):
        context_tokens = tokenizer(context)
        truncate_ratio = max_length / (len(context_tokens["input_ids"]) + 1e-5)
        if truncate_ratio < 1:
            return context[:int(len(context) * truncate_ratio)]
        else:
            return context 
    
    def preprocess_fn(raw_examples):
        """preprocess example strings, mask source part in labels"""
        sources = [
            f"{_truncate_context(context)} TL;DR: " for context in raw_examples['article']
        ]
        examples = [
            f"{_truncate_context(context)} TL;DR: {summary}" for context, summary in zip(raw_examples['article'], raw_examples['highlights'])
        ]
        # left-padded source for validation & testing
        tokenizer.padding_side = "left"
        
        if 'gpt2' in model_name:
            tokenizer.pad_token = tokenizer.eos_token
            
        lp_sources = tokenizer(
            sources,
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100 # mask source
            label[label == tokenizer.pad_token_id] = -100 # mask paddings
        return dict(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            input_ids_lens=sources_tokenized["input_ids_lens"],
            lp_sources=lp_sources["input_ids"])
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "input_ids_lens", "lp_sources"]
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
    model_name,
    max_input_length,
    batch_size,
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
    if 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 1.99k, valid: 619, test: 618")    
        
    def _tokenize_fn(strings):
        """Tokenize a list of strings, memorize source length"""
        tokenizer.padding_side = "right" 
        tokenized_list = [
            tokenizer(
                text,
                max_length=max_input_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            mask.sum().item() - 1 for mask in attention_mask
        ]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )   
    
    def _truncate_context(context, max_length=416):
        context_tokens = tokenizer(context)
        truncate_ratio = max_length / (len(context_tokens["input_ids"]) + 1e-5)
        if truncate_ratio < 1:
            return context[:int(len(context) * truncate_ratio)]
        else:
            return context 
    
    def preprocess_fn(raw_examples):
        """preprocess example strings, mask source part in labels"""
        sources = [
            f"{_truncate_context(' '.join(context))} TL;DR: " for context in raw_examples['source']
        ]
        examples = [
            f"{_truncate_context(' '.join(context))} TL;DR: {summary[0]}" for context, summary in zip(raw_examples['source'], raw_examples['target'])
        ]
        # left-padded source for validation & testing
        tokenizer.padding_side = "left"
        
        if 'gpt2' in model_name:
            tokenizer.pad_token = tokenizer.eos_token
            
        lp_sources = tokenizer(
            sources,
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100 # mask source
            label[label == tokenizer.pad_token_id] = -100 # mask paddings
        return dict(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            input_ids_lens=sources_tokenized["input_ids_lens"],
            lp_sources=lp_sources["input_ids"])
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "input_ids_lens", "lp_sources"]
    )
    dataloader = DataLoader(
        processed_dataset, shuffle=shuffle, 
        collate_fn=default_data_collator, 
        batch_size=batch_size, pin_memory=True,
    )
    return dataloader, tokenizer


def load_piqa(
    split,
    tokenizer_name,
    model_name,
    max_input_length,
    batch_size,
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load piqa dataset
    train: 16.1k, valid: 1.84k, test: 3.08k
    """
    
    dataset = load_dataset(
        "piqa", split=split, 
        keep_in_memory=keep_in_memory
    )
    dataset.cleanup_cache_files()
    if 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 16.1k, valid: 1.84k, test: 3.08k")    
        
    def _tokenize_fn(strings):
        """Tokenize a list of strings, memorize source length"""
        tokenizer.padding_side = "right" 
        tokenized_list = [
            tokenizer(
                text,
                max_length=max_input_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            mask.sum().item() for mask in attention_mask
        ]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )   
    
    def _truncate_context(context, max_length=416):
        context_tokens = tokenizer(context)
        truncate_ratio = max_length / (len(context_tokens["input_ids"]) + 1e-5)
        if truncate_ratio < 1:
            return context[:int(len(context) * truncate_ratio)]
        else:
            return context 
    
    def preprocess_fn(raw_examples):
        """preprocess example strings, mask source part in labels"""
        sources = [
            f"goal:{_truncate_context(goal, 128)}</s>sol1:{_truncate_context(sol1, 128)}</s>sol2:{_truncate_context(sol2, 128)}</s>label:" 
            for goal, sol1, sol2 in zip(raw_examples['goal'], raw_examples['sol1'], raw_examples['sol2'])
        ]
        examples = [
            f"goal:{_truncate_context(goal, 128)}</s>sol1:{_truncate_context(sol1, 128)}</s>sol2:{_truncate_context(sol2, 128)}</s>label:{label}</s>" 
            for goal, sol1, sol2, label in zip(raw_examples['goal'], raw_examples['sol1'], raw_examples['sol2'], raw_examples['label'])
        ]
        
        # left-padded source for validation & testing
        tokenizer.padding_side = "left"
        
        if 'gpt2' in model_name:
            tokenizer.pad_token = tokenizer.eos_token
            
        lp_sources = tokenizer(
            sources,
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100 # mask source
            label[label == tokenizer.pad_token_id] = -100 # mask paddings
        return dict(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            input_ids_lens=sources_tokenized["input_ids_lens"],
            lp_sources=lp_sources["input_ids"])
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "input_ids_lens", "lp_sources"]
    )
    dataloader = DataLoader(
        processed_dataset, shuffle=shuffle, 
        collate_fn=default_data_collator, 
        batch_size=batch_size, pin_memory=True,
    )
    return dataloader, tokenizer


def load_webquestions(
    split,
    tokenizer_name,
    model_name,
    max_input_length,
    batch_size,
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load webquestions dataset
    train: 3.78k, test: 2.03k
    """
    
    dataset = load_dataset(
        "web_questions", split=split, 
        keep_in_memory=keep_in_memory
    )
    dataset.cleanup_cache_files()
    if 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if print_info:
        print("train: 3.78k, test: 2.03k")    
        
    def _tokenize_fn(strings):
        """Tokenize a list of strings, memorize source length"""
        tokenizer.padding_side = "right" 
        tokenized_list = [
            tokenizer(
                text,
                max_length=max_input_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            mask.sum().item() for mask in attention_mask
        ]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )   
    
    def _truncate_context(context, max_length=416):
        context_tokens = tokenizer(context)
        truncate_ratio = max_length / (len(context_tokens["input_ids"]) + 1e-5)
        if truncate_ratio < 1:
            return context[:int(len(context) * truncate_ratio)]
        else:
            return context 
    
    def preprocess_fn(raw_examples):
        """preprocess example strings, mask source part in labels"""
        sources = [
            f"question:{_truncate_context(q, 72)}</s>answer:" for q in raw_examples["question"]
        ]
        
        examples = [
            f"question:{_truncate_context(q, 72)}</s>answer:{_truncate_context(a[0], 64)}</s>" for q, a in zip(raw_examples["question"], raw_examples["answers"])
        ]
        
        # left-padded source for validation & testing
        tokenizer.padding_side = "left"
        
        if 'gpt2' in model_name:
            tokenizer.pad_token = tokenizer.eos_token
            
        lp_sources = tokenizer(
            sources,
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100 # mask source
            label[label == tokenizer.pad_token_id] = -100 # mask paddings
        return dict(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            input_ids_lens=sources_tokenized["input_ids_lens"],
            lp_sources=lp_sources["input_ids"])
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "input_ids_lens", "lp_sources"]
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
    model_name,
    max_input_length,
    batch_size,
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load duorc/SelfRC dataset
    train: 60.7k, valid: 13k, test: 12.6k
    """
    
    def duorc_parser(example):
        n_answer = len(example["answers"])
        return [example["plot"]]*n_answer, [example["question"]]*n_answer, [answer if len(answer) > 0 else "" for answer in example["answers"]]
    
    dataset = load_dataset(
        "duorc", "SelfRC", split=split, 
        keep_in_memory=keep_in_memory
    )
    if 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    else:
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
    
    def _tokenize_fn(strings):
        """Tokenize a list of strings, memorize source length"""
        tokenizer.padding_side = "right" 
        tokenized_list = [
            tokenizer(
                text,
                max_length=max_input_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            mask.sum().item() - 1 for mask in attention_mask
        ]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )   
    
    def _truncate_context(context, max_length=896):
        context_tokens = tokenizer(context)
        truncate_ratio = max_length / (len(context_tokens["input_ids"]) + 1e-5)
        if truncate_ratio < 1:
            return context[:int(len(context) * truncate_ratio)]
        else:
            return context 
              
    
    def preprocess_fn(raw_examples):
        """preprocess example strings, mask source part in labels"""
        sources = [
            f"QUESTION: {question} CONTEXT: {_truncate_context(context)} ANSWER: " for question, context in zip(raw_examples['question'], raw_examples['context'])
        ]
        examples = [
            f"QUESTION: {question} CONTEXT: {_truncate_context(context)} ANSWER: {answer}" for question, context, answer in zip(raw_examples['question'], raw_examples['context'], raw_examples['answer'])
        ]
        
        # left-padded source for validation & testing
        tokenizer.padding_side = "left"
        
        if 'gpt2' in model_name:
            tokenizer.pad_token = tokenizer.eos_token
            
        lp_sources = tokenizer(
            sources,
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100 # mask source
            label[label == tokenizer.pad_token_id] = -100 # mask paddings
        return dict(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            input_ids_lens=sources_tokenized["input_ids_lens"],
            lp_sources=lp_sources["input_ids"])
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "input_ids_lens", "lp_sources"]
    )
    dataloader = DataLoader(
        processed_dataset, shuffle=shuffle, 
        collate_fn=default_data_collator, 
        batch_size=batch_size, pin_memory=True,
    )
    return dataloader, tokenizer
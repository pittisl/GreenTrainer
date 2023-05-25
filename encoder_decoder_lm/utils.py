import numpy as np
import os
from collections import Counter
from tqdm import tqdm


# def count_tokens(dataset, tokenizer):
#     num_input_tokens = []
#     num_output_tokens = []
#     print("###### Analyzing dataset...")
#     for example in tqdm(dataset):
#         input_tokens = tokenizer(example['document'], add_special_tokens=False)
#         output_tokens = tokenizer(example['summary'], add_special_tokens=False)
#         num_input_tokens.append(len(input_tokens['input_ids']))
#         num_output_tokens.append(len(output_tokens['input_ids']))
#     print("###### Number of tokens (input, output) #####")
#     print(f"total: ({np.sum(num_input_tokens)}, {np.sum(num_output_tokens)})")
#     print(f"mean: ({np.mean(num_input_tokens)}, {np.sum(num_output_tokens)})")
#     print(f"max: ({np.max(num_input_tokens)}, {np.sum(num_output_tokens)})")
#     print(f"min: ({np.min(num_input_tokens)}, {np.sum(num_output_tokens)})")

def make_folders(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def flops_counter(model, t_fp, t_dy, t_dw):
    m = []
    for k, (_, param) in enumerate(model.named_parameters()):
        if param.requires_grad == True:
            m.append(1.0)
        else:
            m.append(0.0)
    bp_flops = np.sum(np.maximum.accumulate(m) * t_dy + m * t_dw)
    full_bp_flops = np.sum(t_dy + t_dw)
    fp_flops = t_fp
    bfp_flops = bp_flops + fp_flops
    full_bfp_flops = full_bp_flops + fp_flops
    print("-------------Per Batch FLOPs-----------------")
    print(f"Forward GFLOPs: {fp_flops:.2f}")
    print(f"GreenTrainer Backward GFLOPs: {bp_flops:.2f}")
    print(f"GreenTrainer GFLOPs: {bfp_flops:.2f}")
    print("-------------------------------------")
    print(f"Full Finetuning Backward GFLOPs: {full_bp_flops}")
    print(f"Full Finetuning GFLOPs: {full_bfp_flops:.2f}")
    print("-------------------------------------")
    print(f"GreenTrainer Backward GFLOPs Speedup: {full_bp_flops / bp_flops :.2f}x")
    print(f"GreenTrainer GFLOPs Speedup: {full_bfp_flops / bfp_flops :.2f}x")
    flops = {
        "fp":fp_flops,
        "bp":bp_flops,
        "full_bp":full_bp_flops,
        "bfp":bfp_flops,
        "full_bfp":full_bfp_flops,
    }
    return flops


def generate_response(
    model, 
    train_type,
    tokenizer, 
    inputs_tokens, 
    labels_tokens,
    max_length,
):
    
    if train_type == "prefix_tuning" or train_type == "lora" or train_type == "adalora":
        outputs_tokens = model.generate(
            input_ids=inputs_tokens,
            max_length=max_length, 
            num_beams=4,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
        )
    else:
        outputs_tokens = model.generate(
            inputs=inputs_tokens,
            max_length=max_length, 
            num_beams=4,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
        )
    
    for label in labels_tokens:
        label[label < 0] = tokenizer.pad_token_id

    outputs_text = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs_tokens]
    labels_text =  [tokenizer.decode(x, skip_special_tokens=True) for x in labels_tokens]

    return {
        "labels_text": labels_text, 
        "outputs_text": outputs_text, 
        "labels_tokens": labels_tokens, 
        "outputs_tokens": outputs_tokens,
    }


def compute_squad_metric(tokenizer, predictions, references):
    f1 = exact_match = 0

    for ground_truths, prediction in zip(references, predictions):
        # Remove pad token
        tokens_to_remove = {
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.mask_token_id
        }
        prediction = list(filter(lambda token: token not in tokens_to_remove, prediction))
        ground_truths = list(filter(lambda token: token not in tokens_to_remove, ground_truths))
        f1 += f1_score(prediction, ground_truths)
        exact_match += exact_match_score(prediction, ground_truths)
    
    return {"f1": 100*f1/len(predictions), "EM": 100*exact_match/len(predictions)}


def exact_match_score(prediction_tokens, ground_truth_tokens):
    if len(ground_truth_tokens) == len(prediction_tokens):
        if all(token1 == token2 for token1, token2 in zip(ground_truth_tokens, prediction_tokens)):
            return 1
    return 0


def f1_score(prediction_tokens, ground_truth_tokens):
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
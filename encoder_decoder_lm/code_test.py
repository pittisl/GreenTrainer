import os
import numpy as np
from models import load_text_generation_model
from tensor_flops import compute_tensor_flops, compute_forward_flops


os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_name = "google/flan-t5-small"

model = load_text_generation_model(
    model_name, "full_finetuning",
    output_attentions=False,
)

t_dy, t_dw = compute_tensor_flops(
    model=model,
    model_name=model_name,
    input_length=1024,
    output_length=512,
    batch_size=4,
    draw_figure=True,
)

t_forward = compute_forward_flops(
    model=model,
    model_name=model_name,
    input_length=1024,
    output_length=512,
    batch_size=4,
)

print(f"bp/fp = {np.sum(t_dy + t_dw) / t_forward}")


# summary = layer_summary(model, print_summary=True)
# compute_tensor_flops(summary, num_tokens=512, batch_size=4, draw_figure=True)
    
# ._parameters, in_features, out_features

# for k, (name, param) in enumerate(model.named_parameters()):
#     if param.requires_grad:
#         print(k, name)


# train_loader, tokenizer = load_samsum_for_t5small(
#     split="train", 
#     max_input_length=512, 
#     max_output_length=256,
#     batch_size=8,
#     shuffle=True,
#     keep_in_memory=True,
#     print_info=False,
# )

# print(model)

# # profile_batch_time(
# #     model=model,
# #     dataloader=train_loader,
# # )

# profile_training(
#     model=model,
#     dataloader=train_loader,
#     folder_name="t5small_samsum",
# )


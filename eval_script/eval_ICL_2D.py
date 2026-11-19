import json
import os
import sys
import types
from datetime import datetime

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import evaluation_trainer, get_parser
from dataloader import MetaDataset_Multi_Extended

args = get_parser().parse_args()
print(args)

print("=" * 50)
print("Running UniMedSeg 2D Random Retrieval in Open-Source Sample Mode")
print("=" * 50)

sample_config_path = os.path.join(PROJECT_ROOT, "Liver", "Liver.json")

args.data_dir = PROJECT_ROOT

try:
    with open(sample_config_path, "r") as f:
        data_loading_config = json.load(f)
    print(f"Successfully loaded open-source config from {sample_config_path}")
    print("The current sample task is:", data_loading_config[0]["name"])
except Exception as e:
    raise RuntimeError(f"Failed to load sample config. Error: {e}")
# ----------------------------------------

from UniMedSeg.checkpoints import resolve_checkpoint
from UniMedSeg.lightning_model import LightningModel

checkpoint_path = resolve_checkpoint(args.checkpoint_path, args.checkpoint_index)
model = LightningModel.load_from_checkpoint(
    checkpoint_path, map_location="cpu", strict=True
)

print("!!! Forcing Model Controls !!!")

model.prob_2d = 1
model.num_slices = 1

model._sample_strategy = lambda dataset_name: 0
print("-> Evaluation Mode ON: Model strategy forced to 0 (Random Visual Context only).")


def random_extract_slices(
    self, images, masks, axis, num_samples, context_multiplier, target_g
):
    """Sample each case independently and uniformly among foreground slices."""
    spatial_axis = axis + 1
    reduce_dims = tuple(i for i in range(4) if i != spatial_axis)

    def draw(case, count):
        valid = (masks[case].sum(dim=reduce_dims) > 0).nonzero().flatten()
        if not valid.numel():
            valid = torch.arange(masks[case].shape[spatial_axis], device=masks.device)
        indices = valid[torch.randint(len(valid), (count,), device=valid.device)]
        return tuple(
            x[case].index_select(spatial_axis, indices).movedim(spatial_axis, 0)
            for x in (images, masks)
        )

    target, label = draw(target_g, num_samples)
    supports = [
        draw(g, num_samples * context_multiplier)
        for g in range(len(images))
        if g != target_g
    ]
    contexts = [
        torch.cat(
            [
                item.reshape(num_samples, context_multiplier, *item.shape[1:])
                for item in group
            ],
            dim=1,
        )
        for group in zip(*supports)
    ]
    return target, label, *contexts


model._extract_multi_context_slices = types.MethodType(random_extract_slices, model)
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: ", trainable_params)

dataset_val = MetaDataset_Multi_Extended(
    dataset_dir=args.data_dir,
    skip_resize=args.skip_resize,
    data_loading_config=data_loading_config,
    train_or_val="val",
    train_or_val_split_rate=args.train_or_val_split_rate,
    group_size=args.context_size + 1,
    DDPM="_DDPM" in args.model_name,
    len_divid=args.len_divid,
    cut_length=True,
)

dataset_val.len_divid = 4
batchsize = args.batch_size

dataloader_val = DataLoader(
    dataset_val,
    batch_size=batchsize,
    num_workers=args.workers,
    pin_memory=True,
    persistent_workers=args.workers > 0,
)

trainer = evaluation_trainer(args.device)

target_prompt_types = ["dense"]
final_results = {}

print("=" * 50)
print(f"Starting Individual Prompt Evaluation for: {target_prompt_types}")
print("=" * 50)

for p_type in target_prompt_types:
    print(f"\n>>> Evaluating Prompt Type: [ {p_type} ] ...")

    model.prompt_types_2d = [p_type]
    model.prompt_types_3d = [p_type]
    model.visualize_results = lambda *args, **kwargs: None
    # ==================================================

    trainer.validate(model, dataloaders=dataloader_val)

    metrics_dict = {
        key: (value.item() if isinstance(value, torch.Tensor) else value)
        for key, value in trainer.callback_metrics.items()
    }
    final_results[p_type] = metrics_dict
    print(f">>> Result for {p_type}: {metrics_dict}")

current_time = datetime.now().strftime("%H%M%S")

if hasattr(model, "prob_2d") and model.prob_2d == 1:
    mode_tag = "_2D"
elif hasattr(model, "prob_2d") and model.prob_2d == 0:
    mode_tag = "_3D"
else:
    mode_tag = "_mix"

safe_save_dir = os.path.join(PROJECT_ROOT, "opensource_eval_results")
os.makedirs(safe_save_dir, exist_ok=True)

base_ckpt_name = os.path.basename(checkpoint_path)
new_file_name = base_ckpt_name.replace(
    ".ckpt",
    f"_FLARE22_Sample_context{str(args.context_size)}{mode_tag}_Random_Retrieval_{current_time}_ICL.json",
)

save_filename = os.path.join(safe_save_dir, new_file_name)

print("\nFinal Results Dict:", json.dumps(final_results, indent=4))

with open(save_filename, "w") as f:
    json.dump(final_results, f, indent=4)

print("=" * 50)
print(f"Detailed evaluation metrics saved to:\n{save_filename}")
print("=" * 50)

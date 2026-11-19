# UniMedSeg

UniMedSeg segments medical images with visual examples, interactive prompts, or anatomical text. Both model sizes use the same training and evaluation entry points.

## Environment

Use Linux, Python 3.8, and a CUDA-enabled PyTorch 2.0.1 installation for GPU training. Run the commands below from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Models and weights

Place the weight files in `UniMedSeg/checkpoints/`. Each exported checkpoint contains the backbone, the text projection, and the settings needed to reconstruct that model. Training and evaluation use strict parameter loading.

| Model | Backbone entry point | Weight file | Width | Transformer blocks | Attention heads |
| --- | --- | --- | ---: | ---: | ---: |
| UniMedSeg-S | [primus_icl_s](UniMedSeg/models/PrimusS_ICL.py) | [UniMedSeg-S.ckpt](https://github.com/Lii1228/UniMedSeg/releases/download/s-l-weights-v1/UniMedSeg-S.ckpt) | 432 | 12 | 6 |
| UniMedSeg-L | [primus_icl_l](UniMedSeg/models/PrimusL_ICL.py) | [UniMedSeg-L.ckpt](https://github.com/Lii1228/UniMedSeg/releases/download/s-l-weights-v1/UniMedSeg-L.ckpt) | 864 | 20 | 12 |

Both sizes use the same [PrimusICL implementation](UniMedSeg/models/primus_icl.py). Train L with `--variant L`. To construct its backbone directly in Python, use the following import.

```python
from UniMedSeg.models.PrimusL_ICL import primus_icl_l

backbone = primus_icl_l()
```

The exported files have these SHA-256 checksums. Run `sha256sum UniMedSeg/checkpoints/*.ckpt` to check a download.

```text
2e9545498836bfa5e09925f0d23c74437f19af34e4e3143ed0139cb44276cc47  UniMedSeg-S.ckpt
e223b214fbddb59276c2ca895d679e1ebf8ee2c1e6d1cc6a63eb3b167d33de68  UniMedSeg-L.ckpt
```

The repository includes the cached BiomedBERT embeddings used by the sample text prompts. [BiomedBERT files](https://drive.google.com/drive/folders/1uvEDbAupGgjnFBCjHBD6Fgs1oXtvM2vJ)

## Training

For a quick run on the five included liver cases, initialize from public weights and run one training group and one validation group on a single GPU.

```bash
bash run_training.sh --variant S --init UniMedSeg/checkpoints/UniMedSeg-S.ckpt \
    --data_dir . --task_config Liver/Liver.json --context_size 1 \
    --train_gpus 0 --workers 0 --grad_checkpointing_base 1 \
    --max_epochs 1 --limit_train_batches 1 --limit_val_batches 1
```

For 3D training with weak prompts, set `DATA_ROOT` to the parent directory of your nnU-Net datasets and edit `train_script/train.json` to include the tasks you want to train on. To initialize from base weights, set `BASE_WEIGHTS` to a weight file that matches the model size and pass it with `--init`.

```bash
python -m train_script.train \
    --variant S --init "${BASE_WEIGHTS}" \
    --data_dir "${DATA_ROOT}" --train_gpus '(0,1,2,3,4,5,6,7)' \
    --precision 32 --validation_first 0 --workers 8 --loss_seg DiceCE_loss \
    --lr 0.00001 --lr_decline_patience 120 --skip_resize True --len_divid 50 \
    --context_size 3 --gin_prob 0.05 --grad_checkpointing_base 0 \
    --grad_checkpointing 1 --batch_size 1 --prob_2d 0 --num_slices 1 \
    --prob_a 1 --prob_c 0 --prob_ac 0 --prob_text 0 --prob_at 0 \
    --prob_ct 0 --prob_act 0 --flip_prob 0 --sobel_prob 0
```

To train from scratch, omit `--init`. The following command uses the same training defaults on eight GPUs.

```bash
python -m train_script.train --variant S --data_dir "${DATA_ROOT}" --train_gpus '(0,1,2,3,4,5,6,7)'
```

For UniMedSeg-L, use `--variant L`. When initializing from base weights, select the L weight file.

The launcher honors `--precision 32` as FP32. Use `--precision 16-mixed` for the precision used by the original server launcher. With `--grad_checkpointing_base 0`, large math-backend attention matrices are evaluated in query blocks. Every query still attends to its full set of keys. Set the flag to `1` to let PyTorch select a fused backend when available. Multi-GPU runs use Gloo by default. NCCL is available via `--ddp_backend nccl`.

### Training data

[train.json](train_script/train.json) contains three example tasks from [BraTS 2021](https://www.med.upenn.edu/cbica/brats2021/), [AMOS 2022](https://amos22.grand-challenge.org/) and the [Medical Segmentation Decathlon heart task](https://medicaldecathlon.com/). Keep the entries for datasets you have and add your own to the same JSON array. You can also pass a separate file with `--task_config path/to/tasks.json`.

Prepare each dataset as paired NIfTI images and integer segmentation masks using the [nnU-Net v2 dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). For example, a single-channel renal MRI dataset with two labeled cases has this layout.

```text
DATA_ROOT/RenalMRI/
    dataset.json
    imagesTr/case01_0000.nii.gz
    imagesTr/case02_0000.nii.gz
    labelsTr/case01.nii.gz
    labelsTr/case02.nii.gz
```

In the dataset's own `dataset.json`, declare the image channels in `channel_names`, the integer mask IDs in `labels` (for example, `{"background": 0, "kidney": 1}`), `numTraining` and `file_ending` (here `.nii.gz`). Images and masks for each case must share the same geometry.

To train on kidneys in this example, append the following object to `train.json`.

```json
{
  "name": "Renal MRI Study",
  "directory": "RenalMRI",
  "input": 0,
  "output": "Seg",
  "task": "Seg",
  "task_config": {"foreground_classes": [1]},
  "sample rate": 1,
  "weight": 1
}
```

`name` is the dataset's full name. Set `directory` to its existing folder name under `DATA_ROOT`. If omitted, the loader uses `name` as the folder name. `input` selects the image channel (0 reads `_0000`). `foreground_classes` lists the mask IDs to combine into one target. To sample foreground labels at random, use `"foreground_classes": "random"` with `"remove": [0]` to exclude background. `sample rate` controls how often the loader selects the task relative to the other entries.

Provide at least two labeled cases per dataset. The loader sorts case IDs, uses the first 90% for training (rounded down), and reserves the rest for validation. Each group contains one target and `context_size` support volumes from the same task and split. With `--skip_resize True`, supply aligned 128 × 128 × 128 volumes. Use `--skip_resize False` to crop, pad and resize during loading. Training uses 3D batches by default. Set `--prob_2d` to the desired fraction of 2D batches.

### Checkpoints and resuming

CSV metrics, the best validation checkpoint, and `last.ckpt` are written under `runs/UniMedSeg-{S,L}/version_*/`. To restore model, optimizer, scheduler and epoch state, reuse the training arguments and supply `--resume /path/to/last.ckpt` in place of `--init`. Set `--max_epochs` to the desired total epoch count. Released weight files support initialization and inference.

## Evaluation

The first command runs the Brain text sample with L. The other entry points use the Liver examples for visual in-context or interactive evaluation.

```bash
bash run_inference.sh UniMedSeg/checkpoints/UniMedSeg-L.ckpt
python eval_script/eval_ICL.py --checkpoint_path UniMedSeg/checkpoints/UniMedSeg-S.ckpt
python eval_script/eval_ICL_2D.py --checkpoint_path UniMedSeg/checkpoints/UniMedSeg-S.ckpt
python eval_script/eval_interactive.py --checkpoint_path UniMedSeg/checkpoints/UniMedSeg-L.ckpt
```

Evaluation writes JSON metrics to `opensource_eval_results/`.

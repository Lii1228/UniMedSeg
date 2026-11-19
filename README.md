# UniMedSeg

UniMedSeg segments medical images with visual examples, interactive prompts, or anatomical text. Both model sizes use the same training and evaluation entry points.

## Environment

Use Linux, Python 3.8, and a CUDA-enabled PyTorch 2.0.1 installation for GPU training. Run the commands below from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Models and weights

Place the weight files in `UniMedSeg/checkpoints/`. Each exported checkpoint contains the backbone, the text projection, and the settings needed to reconstruct that model. Training and evaluation use strict parameter loading.

| Model | Weight file | Width | Transformer blocks | Attention heads |
| --- | --- | ---: | ---: | ---: |
| UniMedSeg-S | [UniMedSeg-S.ckpt](https://github.com/Lii1228/UniMedSeg/releases/download/s-l-weights-v1/UniMedSeg-S.ckpt) | 432 | 12 | 6 |
| UniMedSeg-L | [UniMedSeg-L.ckpt](https://github.com/Lii1228/UniMedSeg/releases/download/s-l-weights-v1/UniMedSeg-L.ckpt) | 864 | 20 | 12 |

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

For 3D training with weak prompts, set `DATA_ROOT` to the preprocessed training collection. This example initializes from the public S weights.

```bash
python -m train_script.train \
    --variant S --init UniMedSeg/checkpoints/UniMedSeg-S.ckpt \
    --data_dir "${DATA_ROOT}" --train_gpus '(0,1,2,3,4,5,6,7)' \
    --precision 32 --validation_first 0 --workers 8 --loss_seg DiceCE_loss \
    --lr 0.00001 --lr_decline_patience 120 --skip_resize True --len_divid 50 \
    --context_size 3 --gin_prob 0.05 --grad_checkpointing_base 0 \
    --grad_checkpointing 1 --batch_size 1 --prob_2d 0 --num_slices 1 \
    --prob_a 1 --prob_c 0 --prob_ac 0 --prob_text 0 --prob_at 0 \
    --prob_ct 0 --prob_act 0 --flip_prob 0 --sobel_prob 0
```

A checkpoint directory uses lexicographic filename order, matching the research launcher. An explicit filename selects exactly one checkpoint. For L fine-tuning, replace the S variant and initialization arguments with `--variant L --init UniMedSeg/checkpoints/UniMedSeg-L.ckpt`.

The launcher honors `--precision 32` as FP32. Use `--precision 16-mixed` for the precision used by the original server launcher. With `--grad_checkpointing_base 0`, large math-backend attention matrices are evaluated in query blocks. Every query still attends to its full set of keys. Set the flag to `1` to let PyTorch select a fused backend when available. Multi-GPU runs use Gloo by default. NCCL is available via `--ddp_backend nccl`.

### Training data

Both models use the 473 tasks in [train.json](train_script/train.json). For a custom dataset, pass `--task_config` with a task list in the same format as [the Liver example](Liver/Liver.json). Each task specifies an input channel, foreground labels, and a sampling weight (`sample rate`).

Omit `--init` to train from random initialization. Training uses 3D batches by default. Set `--prob_2d` to the desired fraction of 2D batches. Run `python -m train_script.train --help` for the available options.

A dataset named `Liver` is read from the following paths.

```text
DATA_ROOT/Liver/
    dataset.json
    imagesTr/CASE_0000.nii.gz
    labelsTr/CASE.nii.gz
```

The loader sorts case IDs and reserves the last 10% for validation, rounding the training count down. Each training group contains one target and `context_size` support volumes sampled with replacement from the same task and split. With `--skip_resize True`, supply aligned 128 x 128 x 128 images and labels. Use `--skip_resize False` to crop, pad and resize during loading. The bundled Brain case is for text evaluation. The Liver sample has cases in both training and validation splits.

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

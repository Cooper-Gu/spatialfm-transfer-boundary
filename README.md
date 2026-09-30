# SpatialFM transfer-boundary study

This repository contains the reproducible code and compact report bundle for the study **“Beyond local gains: cross-donor transfer boundaries of spatial transcriptomic representations.”** It evaluates frozen spatial transcriptomic representations under donor-heldout, cross-source and continuous spatial readouts. The repository documents the tested models and evidence boundary.

This GitHub copy uses descriptive file, directory and model-mode names. The [Zenodo protocol archive](https://doi.org/10.5281/zenodo.23001983) is the authoritative frozen record. SHA-256 values in the preregistration manifests and locks refer to that original archive and do not verify the edited files in this copy. To rerun analyses with local reports, place them under the descriptive paths used by the scripts.

## What is included

- `preregistration/`: the archived protocol, analysis contract, environment lock and confirmatory evaluators.
- `model_transformer.py`: the dense and block-sparse gene-token Transformer implementations used in the architecture audits.
- `train_transformer_ddp.py`: the multi-GPU masked-gene pretraining entry point, including cross-rank negatives, gradient accumulation, covariance/whitening options and PyTorch SDP attention controls.
- `train_maskaware_program.py`: the development MLP representation trainer.
- `analysis/`: dataset-cache builders, donor-heldout evaluators, spatial diagnostics and evidence-table exporters.
- `artifacts/`: compact JSON reports and tables supporting the reported summaries. Raw matrices, images, checkpoints and downloaded public datasets are intentionally excluded.

## Environment

The original runs used Python 3.12, NumPy, PyTorch, scikit-learn, SciPy, h5py, pandas and anndata. A CUDA installation is needed for model training; the evaluation scripts can run on CPU for small caches. The exact locked environment used for the preregistered analysis is recorded in `preregistration/environment_lock.json`.

Install the common packages in a clean environment, then run the smoke test before using real data:

```bash
python -m pip install -r requirements.txt
python preregistration/execute_reviewer_fullclass.py \
  --smoke-test \
  --out /tmp/spatialfm_smoke.json
```

The smoke test is an implementation check and must not be reported as a biological result.

## Reproducing the donor-heldout endpoint

The confirmatory evaluator consumes one `.npz` file per spatial slice. Each file must provide `E`, `genes`, `y`, `panel` and platform metadata, with a consistent gene order. A donor map can be supplied separately. The exact protocol and endpoint are specified in `preregistration/PREREG_SFM_transfer_data_condition_gate.md` and `preregistration/analysis_contract.json`.

Example:

```bash
python preregistration/execute_reviewer_fullclass.py \
  --cache-dir /path/to/spatiallibd_npz \
  --checkpoint /path/to/frozen_checkpoint.pt \
  --device cuda \
  --out reports/dlpfc_fullclass.json
```

Do not use held-out donor labels for checkpoint selection, panel selection, PCA fitting or probe fitting. The fixed seven-class ontology is `{L1, L2, L3, L4, L5, L6, WM}`; absent classes remain in the score space and are reported explicitly.

## Training a Transformer audit checkpoint

The training script is designed for a preprocessed `.npz` region cache. A four-GPU launch is:

```bash
torchrun --standalone --nproc_per_node=4 train_transformer_ddp.py \
  --root /data/spatial_foundation_model_data/pasa_fm/data \
  --npz /data/spatial_foundation_model_data/pasa_fm/data/processed/regions_32um.npz \
  --model-mode sparse_foundation \
  --layers 8 \
  --batch 16 \
  --grad-accum-steps 4 \
  --cross-rank-negatives \
  --nonzero-aware-loss \
  --out /data/spatial_foundation_model_data/pasa_fm/outputs/checkpoint.json
```

The script enables PyTorch scaled-dot-product attention backends when CUDA is available. Whether the fast Flash attention kernel is actually selected depends on the installed PyTorch/CUDA hardware combination; the run log records the backend state. The repository does not include credentials or model weights. Historic report records include paths from the original research environment; provide paths for your own data and checkpoints when running the scripts.

## Citation

The study protocol is publicly archived at [Zenodo DOI 10.5281/zenodo.23001983](https://doi.org/10.5281/zenodo.23001983). The manuscript and figure legends in the local research package provide the full evidence scope and limitations.

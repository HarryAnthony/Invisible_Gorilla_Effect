# Data directory

This directory includes download instructions, expected on-disk layout and **manual artefact annotations** used in the paper benchmarks. The raw image data for CheXpert, ISIC and MVTec-AD are **not** included in this repository and must be downloaded separately under the terms of each dataset licence.

Each dataset is wired into the codebase through a config module under `source/config/`:

| Dataset | CLI name | Config module | Subdirectory |
|---------|----------|---------------|--------------|
| CheXpert | `chexpert` | `source/config/chexpert.py` | [`CheXpert/`](CheXpert/README.md) |
| ISIC | `ISIC` | `source/config/ISIC.py` | [`ISIC/`](ISIC/README.md) |
| MVTec-AD | `MVTec` | `source/config/MVTec.py` | [`MVTec/`](MVTec/README.md) |

Preset experiment splits (e.g. `setting1`, `setting2`) are defined in those config files via `dataset_selection_settings` and `OOD_selection_settings`. See the [main README](../README.md) for training and evaluation usage.

---

## CheXpert

**Source:** [CheXpert-v1.0-small](https://stanfordmlgroup.github.io/competitions/chexpert/) (Stanford ML Group)

**Purpose in this repo:** Chest X-ray multilabel classification with OOD benchmarks based on pathology splits, demographics, and support-device artefacts.

### Download and layout

1. Download **CheXpert-v1.0-small** and extract it under `data/CheXpert/`.
2. The config expects:
   - **Metadata CSVs:** `data/CheXpert/CheXpert-v1.0-small/train.csv`, `valid.csv`
   - **Images:** paths in the CSVs are relative to `loader_root` in `source/config/chexpert.py` (default `data/CheXpert/`)

Typical extracted structure:

```
data/CheXpert/CheXpert-v1.0-small/
├── train.csv
├── valid.csv
├── train/
└── valid/
```

Update `root` and `loader_root` in `source/config/chexpert.py` if your paths differ.

### Label space

Fourteen CheXpert competition labels are supported, including `No Finding`, `Cardiomegaly`, `Pleural Effusion`, `Fracture`, `Support Devices`, and others (full list in `source/config/chexpert.py`).

### Annotations

Manual annotations shipped with this repo live under `data/CheXpert/annotations/`:

| File | Description |
|------|-------------|
| `no_support_device.txt` | Frontal chest X-rays with **no support devices** visibly obscuring the chest (lines, tubes, pacemakers, hardware, etc.) |

These annotations were produced for prior work on Mahalanobis OOD detection.

---

## ISIC

**Source:** [ISIC Archive / challenge data](https://challenge.isic-archive.com/data/)

**Purpose in this repo:** Dermoscopy lesion classification (e.g. malignant vs benign) with OOD benchmarks built from **real imaging artefacts** (ink markings, colour charts) identified with manual annotation.

### Download and layout

1. Download ISIC images and metadata from the ISIC Archive (see [`ISIC/download_ISIC_datasets.sh`](ISIC/download_ISIC_datasets.sh) for example `isic-cli` collection IDs used in this project).
2. Build or place a **processed** dataset with:
   - `train.csv`, `valid.csv` (and optionally `test.csv`) at the path set by `root` in `source/config/ISIC.py`
   - Image files reachable from `loader_root` + `Path` column entries

**Important:** Before running experiments, Set `root` and `loader_root` to your own ISIC directory (for example `data/ISIC/` after processing) in `source/config/ISIC.py`.

Example expected CSV columns include image `Path` and per-diagnosis label columns that are mapped to classes such as `malignant` and `benign` during dataset selection.

### Preset settings (summary)

| Setting | ID task (training) | OOD evaluation (`different_class`) |
|---------|-------------------|-------------------------------------|
| `setting1` | Malignant vs benign; training restricted to artefact-free images | Ink annotation images |
| `setting2` | Malignant vs benign; training restricted to artefact-free images | Colour-chart images |

Both use **5-fold cross-validation** on ID data.

### New Manual annotations

All paths below are relative to `data/ISIC/annotations/`:

| Folder | Description |
|--------|-------------|
| `Training data/training_data.txt` | Images **without** coloured artefacts (rulers, ink, colour charts), used to restrict ID training and avoid shortcut learning |
| `Ink_annotations/` | Images with **ink** artefacts; `ink_annotation.txt` is the full set; colour-specific lists: `black.txt`, `green.txt`, `purple.txt`, `red.txt` |
| `Colour_chart/` | Images with **colour-chart** artefacts; `colour_chart.txt` is the full set; per-colour lists: `black.txt`, `blue.txt`, `green.txt`, `orange.txt`, `red.txt`, `yellow.txt`, `white.txt` |
| `Colour_chart/below_10p_coverage/` | Colour-chart images where the artefact covers **less than 10%** of image pixels (subset by colour) |

Selection logic is implemented in [`source/dataloaders/ISIC_dataloader.py`](../source/dataloaders/ISIC_dataloader.py).


---

## MVTec-AD

**Source:** [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

**Purpose in this repo:** Industrial defect classification on specific defect categories, with OOD benchmarks based on **colour (ink) artefact** annotations on test images.

### Download and layout

1. Download MVTec-AD from MVTec Software GmbH.
2. Prepare `train.csv` and `valid.csv` under `data/MVTec/` (paths set by `root` and `loader_root` in `source/config/MVTec.py`).
3. CSVs should include columns used by this codebase (including `Path`, `class`, and `Object` for category filtering).

Default config paths:

```python
root = 'data/MVTec/'
loader_root = 'data/MVTec'
```

### Preset settings (summary)

| Setting | Object | ID classes (training) | OOD evaluation |
|---------|--------|----------------------|----------------|
| `setting1` | Pill | good, contamination, crack, faulty_imprint, scratch | Colour artefact (`color`) |
| `setting2` | Metal nut | good, bent, flip, scratch | Colour artefact (`color`) |

Both use **5-fold cross-validation** on ID data.

### New Manual annotations

Under `data/MVTec/annotations/`, text files list **image indices** with colour ink artefacts:

| Folder | Files | Colours annotated |
|--------|-------|-------------------|
| `Pill/` | `red`, `yellow` | Red and yellow ink on Pill images |
| `Metal nut/` | `black`, `blue` | Black and blue ink on Metal nut images |

These annotations support OOD evaluation when colour artefacts are treated as out-of-distribution relative to the defect classes seen during training.

Further details: [`MVTec/README.md`](MVTec/README.md)

---

## Manual annotations (general)

A contribution of the associated paper is **public OOD benchmarks** built from manually annotated real artefacts on ISIC and MVTec-AD, and support-device / no-support-device lists on CheXpert. 

**Disclaimer:** Annotations were made by visual inspection for research use and were **not validated by clinical or industrial domain experts**. They are intended for ML research only. If you use these annotations, please cite the paper (see [main README](../README.md)).

<p align="center">
  <img src="../figures/Annotation_summary_ige.jpg" width="600" />
</p>
<p align="center"><i>Visualisation of manual annotations of artefacts by colour.</i></p>


---

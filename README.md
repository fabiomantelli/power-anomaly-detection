# Unsupervised Anomaly Detection for 50 Hz Synchrophasor Networks

Detects anomalies in PMU (Phasor Measurement Unit) data from the **Chile** and **Argentina** 50 Hz transmission grids using [IBM Granite TSPulse](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1) — a zero-shot, dual-domain (time + spectral) time-series foundation model ranked #1 on the TSB-AD benchmark.

---

## Overview

| | |
|---|---|
| **Data source** | openHistorian 2 (remote server via VPN) |
| **Sampling rate** | 60 samples/sec per PMU |
| **PMUs** | 5–20 units (Chile + Argentina) |
| **Channels per PMU** | 8 (Va/Vb/Vc magnitude & angle, freq, ROCOF) |
| **Period** | 2 years (2022–2023) |
| **Model** | TSPulse — 1M parameters, zero-shot |
| **Training hardware** | vast.ai H100 80 GB |

---

## Tech Stack

- **Python 3.11**, PyTorch 2.5, HuggingFace Transformers
- **IBM Granite TSPulse** (`granite-tsfm`) — time-series foundation model
- **Apache Parquet** (PyArrow) — partitioned columnar storage
- **openHistorian 2** — PMU data historian
- **vast.ai** — on-demand GPU cloud (H100)

---

## Prerequisites

### Local environment (data export)

```bash
pip install openhistorian pyarrow>=17 pandas numpy tqdm
```

### Cloud environment — vast.ai (training & inference)

Recommended Docker image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`

```bash
conda create -n pmu-ai python=3.11 && conda activate pmu-ai
pip install -r requirements.txt
pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.3.1"
```

---

## Pipeline

### Phase 1 — Setup (run once)

#### 1. Map available signals from openHistorian

Connect via VPN and run the discovery script:

```bash
python scripts/01_list_signals.py --host 192.168.x.x
```

This creates/updates `configs/signals.json` with all PMU point IDs.

**Manual step:** open `scripts/01_list_signals.py` and fill in the `DEVICE_COUNTRY` dictionary with each PMU's acronym and country. Example:

```python
DEVICE_COUNTRY = {
    "UCHILE01": "Chile",
    "USACH01":  "Chile",
    "UBA01":    "Argentina",
}
```

---

### Phase 2 — Data Export (run locally with VPN active)

#### 2. Export openHistorian data to Parquet

Output format: `D:/openHistorian_export/year=YYYY/month=MM/day=DD/data.parquet`

```bash
# Test with 1 day before running the full range:
python scripts/02_export_to_parquet.py \
    --host  192.168.x.x \
    --start 2022-01-01 \
    --end   2022-01-02 \
    --output D:/openHistorian_export

# Full export (can be split by semester and run in parallel):
python scripts/02_export_to_parquet.py \
    --host  192.168.x.x \
    --start 2022-01-01 \
    --end   2024-01-01 \
    --output D:/openHistorian_export
```

> The script automatically resumes from the last successfully exported day on failure.
>
> Estimated volume (10 PMUs × 8 channels, 2 years): **15–25 GB** in Parquet Snappy.
> To speed things up, run two processes in parallel with distinct semesters.

**Verify:**

```python
import pyarrow.dataset as ds
d = ds.dataset("D:/openHistorian_export", format="parquet")
print(d.count_rows())  # ~302 billion rows for 10 PMUs / 2 years
print(d.schema)
```

#### 3. Pivot from long to wide format

Groups point IDs as columns. Output: `D:/export_wide/year=YYYY/month=MM/data.parquet`

```bash
python scripts/03_pivot_wide.py \
    --input  D:/openHistorian_export \
    --output D:/export_wide
```

> To process a specific month only: `--year 2022 --month 3`

**Verify:**

```python
import pandas as pd
df = pd.read_parquet("D:/export_wide/year=2022/month=01/data.parquet")
print(df.shape)    # (N_timestamps, N_channels + 1)
print(df.columns)  # timestamp_us + one column per point ID
```

---

### Phase 3 — Training (run on vast.ai instance)

#### 4. Transfer data to vast.ai

Provision an H100 80 GB instance on [vast.ai](https://vast.ai) and get the SSH IP/port.

```bash
rsync -avz --progress -e "ssh -p <VAST_PORT>" \
      D:/export_wide/ root@<VAST_IP>:/workspace/pmu_data/
```

> For volumes > 100 GB, consider vast.ai Persistent Storage.

#### 5. Fit the preprocessor (per-channel Z-score)

Use one month of known-normal operation to estimate mean and standard deviation per channel:

```bash
python scripts/04_fit_preprocessor.py \
    --data-dir /workspace/pmu_data \
    --year 2022 --month 3 \
    --output   checkpoints/pmu_preprocessor
```

**Expected log values:**
- `FREQ` channel: mean ≈ 50.0 Hz, std ≈ 0.05–0.2 Hz
- `VPHM` channels: mean ≈ 1.0 pu, std ≈ 0.01–0.05 pu

#### 6. Fine-tune TSPulse on normal data

The model learns to reconstruct normal series. Anomalies surface as high reconstruction loss.

```bash
python scripts/05_finetune_tspulse.py \
    --data-glob    "/workspace/pmu_data/year=2022/month=*/data.parquet" \
    --signals      configs/signals.json \
    --preprocessor checkpoints/pmu_preprocessor \
    --output       checkpoints/tspulse_pmu_finetuned \
    --epochs       10 \
    --batch-size   64
```

> Checkpoints are saved per epoch under `checkpoints/tspulse_pmu_finetuned/epoch_XX/`.
> Monitor loss with TensorBoard: `tensorboard --logdir checkpoints/tspulse_pmu_finetuned`

**Expected:** reconstruction loss decreases each epoch.

---

### Phase 4 — Anomaly Detection (inference)

#### 7. Run the scoring pipeline

```bash
# Zero-shot (no fine-tuning, for initial validation):
python scripts/06_anomaly_detection.py \
    --data-glob "/workspace/pmu_data/year=2023/month=01/data.parquet" \
    --model     ibm-granite/granite-timeseries-tspulse-r1 \
    --preprocessor checkpoints/pmu_preprocessor \
    --output    results/scores_zeroshot_2023_01.parquet

# Fine-tuned model:
python scripts/06_anomaly_detection.py \
    --data-glob "/workspace/pmu_data/year=2023/month=*/data.parquet" \
    --model     checkpoints/tspulse_pmu_finetuned/model \
    --preprocessor checkpoints/pmu_preprocessor \
    --output    results/scores_finetuned_2023.parquet \
    --threshold 3.0
```

Output contains `timestamp_us`, `anomaly_score`, and `anomaly_flag` (`1` = anomaly detected).

**Expected behavior:** `anomaly_flag` near 0% during normal operation. Grid events (frequency dips, voltage imbalance) should produce clear spikes in `anomaly_score`.

---

## Repository Structure

```
power-anomaly-detection/
  configs/
    signals.json              # point_id → signal/PMU/country mapping (generated by 01)
  scripts/
    01_list_signals.py        # Signal discovery from openHistorian
    02_export_to_parquet.py   # Daily export (long format)
    03_pivot_wide.py          # Long → wide pivot by month
    04_fit_preprocessor.py    # Per-channel Z-score fitting
    05_finetune_tspulse.py    # TSPulse fine-tuning
    06_anomaly_detection.py   # Inference and anomaly scoring
  src/
    pmu_dataset.py            # PMUSlidingWindowDataset (streaming IterableDataset)
  checkpoints/
    pmu_preprocessor/         # Per-channel scalers (generated by 04)
    tspulse_pmu_finetuned/    # Fine-tuned model + per-epoch checkpoints (generated by 05)
  results/                    # Anomaly score Parquet files (generated by 06)
  requirements.txt
```

---

## PMU Channels

| Channel | Signal | openHistorian type |
|---|---|---|
| `va_mag` | Phase A voltage — magnitude (pu) | `VPHM` |
| `vb_mag` | Phase B voltage — magnitude (pu) | `VPHM` |
| `vc_mag` | Phase C voltage — magnitude (pu) | `VPHM` |
| `va_ang` | Phase A voltage — angle (degrees) | `VPHA` |
| `vb_ang` | Phase B voltage — angle (degrees) | `VPHA` |
| `vc_ang` | Phase C voltage — angle (degrees) | `VPHA` |
| `freq`   | System frequency (Hz) | `FREQ` |
| `dfreq`  | ROCOF — df/dt (Hz/s) | `ROCOF` |

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Storage format | Partitioned Parquet (year/month/day) | Efficient streaming, columnar layout for ML workloads |
| Export compression | Snappy | Fast writes during export |
| Wide format compression | Zstd level 3 | Better ratio for correlated float channels |
| Context window | 512 samples (~8.5s at 60 Hz) | TSPulse default; covers typical grid transients |
| Stride | 60 samples (1s) | 1-second resolution for operational alerting |
| Model | TSPulse (IBM Granite) | #1 on TSB-AD benchmark, zero-shot, 1M params |
| Training strategy | Fine-tune on normal data only | No labeled anomalies needed — learns normal distribution |
| Cloud hardware | vast.ai H100 80 GB | Low cost vs. AWS/GCP for GPU-intensive workloads |

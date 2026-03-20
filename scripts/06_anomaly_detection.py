"""
06_anomaly_detection.py
Pipeline de detecção de anomalias usando o TSPulse fine-tuned.

Produz um arquivo Parquet com anomaly_score e flag por timestamp.
Suporta modo zero-shot (modelo base) e fine-tuned.

Uso:
    python scripts/06_anomaly_detection.py \
        --data-glob "/workspace/pmu_data/year=2023/month=01/data.parquet" \
        --signals   configs/signals.json \
        --model     checkpoints/tspulse_pmu_finetuned/model \
        --preprocessor checkpoints/pmu_preprocessor \
        --output    results/anomaly_scores_2023_01.parquet \
        [--threshold 3.0] [--batch-size 64]

Modo zero-shot (sem fine-tuning):
    --model ibm-granite/granite-timeseries-tspulse-r1
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

CONTEXT_LENGTH   = 512   # amostras por janela
STRIDE           = 60    # 1 segundo a 60 Hz
AGGREGATION_LEN  = 60    # janela de agregação de score (1s)
SMOOTHING_LEN    = 30    # suavização 0.5s


def run_pipeline(
    data_glob: str,
    feature_cols: list[str],
    model_path: str,
    preprocessor_path: str,
    output_path: Path,
    threshold_sigma: float,
    batch_size: int,
) -> None:
    try:
        from tsfm_public import TimeSeriesPreprocessor
        from tsfm_public.models.tspulse import TSPulseForReconstruction
        from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
            TimeSeriesAnomalyDetectionPipeline,
        )
        from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods
    except ImportError as e:
        log.error("Dependência ausente: %s. Instale granite-tsfm.", e)
        raise

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from pmu_dataset import PMUSlidingWindowDataset

    preprocessor = TimeSeriesPreprocessor.from_pretrained(preprocessor_path)
    log.info("Preprocessor carregado de %s", preprocessor_path)

    model = TSPulseForReconstruction.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    log.info("Modelo carregado de %s (device=%s)", model_path, device)

    # Pipeline TSFM para scoring de anomalias
    pipeline = TimeSeriesAnomalyDetectionPipeline(
        model,
        timestamp_column="timestamp_us",
        target_columns=feature_cols,
        prediction_mode=[
            AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
            AnomalyScoreMethods.PREDICTIVE.value,
        ],
        aggregation_length=AGGREGATION_LEN,
        aggr_function="max",
        smoothing_length=SMOOTHING_LEN,
        least_significant_scale=0.01,
        device=device,
    )

    # Carrega dados de teste em memória (adaptável a streaming para volumes grandes)
    import glob
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo: {data_glob}")
    log.info("Processando %d arquivo(s)...", len(files))

    all_results: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_parquet(path, columns=["timestamp_us"] + feature_cols)
        df = df.sort_values("timestamp_us").reset_index(drop=True)
        # Substituir NaN por 0 (PMU offline)
        df[feature_cols] = df[feature_cols].fillna(0.0)

        log.info("  Arquivo %s: %d amostras", path, len(df))
        results = pipeline(df, batch_size=batch_size)
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)

    # Determinar threshold baseado nos scores calculados
    score_col = "anomaly_score" if "anomaly_score" in combined.columns else combined.columns[-1]
    scores = combined[score_col].values
    mean_score = float(np.nanmean(scores))
    std_score  = float(np.nanstd(scores))
    threshold  = mean_score + threshold_sigma * std_score
    log.info(
        "Score: mean=%.4f std=%.4f | threshold (%.1fσ)=%.4f",
        mean_score, std_score, threshold_sigma, threshold,
    )

    combined["anomaly_flag"] = (combined[score_col] > threshold).astype(np.int8)

    # Estatísticas de detecção
    n_flagged = int(combined["anomaly_flag"].sum())
    n_total   = len(combined)
    log.info(
        "Amostras flagradas: %d / %d (%.2f%%)",
        n_flagged, n_total, 100.0 * n_flagged / max(n_total, 1),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, output_path, compression="zstd", compression_level=3)
    log.info("Resultados salvos em %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Detecção de anomalias em dados de PMU com TSPulse")
    parser.add_argument("--data-glob",    required=True,                                   help="Glob dos Parquet wide de teste")
    parser.add_argument("--signals",      default="configs/signals.json",                  help="Arquivo de sinais")
    parser.add_argument("--model",        default="checkpoints/tspulse_pmu_finetuned/model", help="Modelo TSPulse (local ou HF hub)")
    parser.add_argument("--preprocessor", default="checkpoints/pmu_preprocessor",          help="Preprocessor salvo")
    parser.add_argument("--output",       default="results/anomaly_scores.parquet",         help="Arquivo de saída")
    parser.add_argument("--threshold",    type=float, default=3.0,                          help="Threshold em sigmas acima da média")
    parser.add_argument("--batch-size",   type=int,   default=64,                           help="Batch size para inferência")
    args = parser.parse_args()

    with open(args.signals, encoding="utf-8") as f:
        signals_cfg = json.load(f)
    feature_cols = [str(pid) for pid in signals_cfg["all_point_ids"]]

    run_pipeline(
        data_glob=args.data_glob,
        feature_cols=feature_cols,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        output_path=Path(args.output),
        threshold_sigma=args.threshold,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

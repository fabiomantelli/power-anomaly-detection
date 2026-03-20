"""
04_fit_preprocessor.py
Ajusta o TimeSeriesPreprocessor (Z-score por canal) em dados de operação normal.

Recomendação: usar 1-4 semanas de dados sem eventos conhecidos para estimar
média e desvio padrão de cada canal (80 canais para 10 PMUs × 8 sinais).

Uso:
    python scripts/04_fit_preprocessor.py \
        --data-dir D:/export_wide \
        --signals  configs/signals.json \
        --output   checkpoints/pmu_preprocessor \
        [--year 2022] [--month 3]

Verificação pós-ajuste:
    - Canal freq: média ≈ 50.0 Hz, std ≈ 0.05-0.2 Hz
    - Canais VPHM: média ≈ 1.0 pu, std ≈ 0.01-0.05 pu
    - Canais VPHA: média ≈ 0°/120°/240° (relativo) ou ≈ 0° se diferença entre fases
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_normal_data(
    data_dir: Path,
    point_id_list: list[int],
    year: int,
    month: int,
) -> pd.DataFrame:
    """Carrega dados wide de um mês específico."""
    parquet_file = data_dir / f"year={year}" / f"month={month:02d}" / "data.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {parquet_file}")

    log.info("Carregando %s ...", parquet_file)
    cols = ["timestamp_us"] + [str(pid) for pid in point_id_list]
    df = pd.read_parquet(parquet_file, columns=cols)
    log.info("  Shape: %s", df.shape)
    return df


def fit_and_save_preprocessor(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    context_length: int = 512,
) -> None:
    """Ajusta o TSFMPreprocessor e salva em disco."""
    try:
        from tsfm_public import TimeSeriesPreprocessor
    except ImportError:
        log.error("granite-tsfm não instalado. Execute: pip install granite-tsfm")
        raise

    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp_us",
        target_columns=feature_cols,
        context_length=context_length,  # 512 amostras = ~8.5s a 60 Hz
        prediction_length=0,
        scaling=True,
        scaler_type="standard",         # Z-score por canal
    )
    log.info("Ajustando preprocessor em %d amostras × %d canais ...", len(df), len(feature_cols))
    tsp.train(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    tsp.save_pretrained(str(output_dir))
    log.info("Preprocessor salvo em %s", output_dir)


def verify_scaler_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    signals_meta: dict,
) -> None:
    """Imprime estatísticas de verificação por tipo de sinal."""
    pid_to_meta = signals_meta.get("point_id_to_meta", {})
    stats_by_type: dict[str, list] = {}

    for col in feature_cols:
        sig_type = pid_to_meta.get(col, {}).get("signal_type", "UNKNOWN")
        series = df[col].dropna()
        if len(series) == 0:
            continue
        stats_by_type.setdefault(sig_type, []).append({
            "col": col,
            "mean": float(series.mean()),
            "std":  float(series.std()),
            "min":  float(series.min()),
            "max":  float(series.max()),
            "nan_pct": float(df[col].isna().mean() * 100),
        })

    for sig_type, stats_list in sorted(stats_by_type.items()):
        means = [s["mean"] for s in stats_list]
        stds  = [s["std"]  for s in stats_list]
        nans  = [s["nan_pct"] for s in stats_list]
        log.info(
            "[%s] n=%d | mean=%.4f±%.4f | std=%.4f±%.4f | nan=%.1f%%",
            sig_type, len(stats_list),
            np.mean(means), np.std(means),
            np.mean(stds),  np.std(stds),
            np.mean(nans),
        )

    # Alertas específicos para rede 50 Hz
    freq_cols = [
        col for col in feature_cols
        if pid_to_meta.get(col, {}).get("signal_type") == "FREQ"
    ]
    for col in freq_cols:
        mean_freq = df[col].mean()
        if not (49.5 <= mean_freq <= 50.5):
            log.warning(
                "Canal %s: freq média = %.3f Hz (esperado ~50 Hz) — verificar unidade/escala",
                col, mean_freq,
            )


def main():
    parser = argparse.ArgumentParser(description="Ajusta preprocessor em dados normais de PMU")
    parser.add_argument("--data-dir",        default="D:/export_wide",          help="Diretório wide Parquet")
    parser.add_argument("--signals",         default="configs/signals.json",    help="Arquivo de sinais")
    parser.add_argument("--output",          default="checkpoints/pmu_preprocessor", help="Diretório de saída")
    parser.add_argument("--year",            type=int, default=2022,            help="Ano dos dados normais")
    parser.add_argument("--month",           type=int, default=3,               help="Mês dos dados normais")
    parser.add_argument("--context-length",  type=int, default=512,             help="Comprimento da janela de contexto")
    args = parser.parse_args()

    with open(args.signals, encoding="utf-8") as f:
        signals_cfg = json.load(f)

    point_id_list: list[int] = signals_cfg["all_point_ids"]
    feature_cols = [str(pid) for pid in point_id_list]

    df = load_normal_data(Path(args.data_dir), point_id_list, args.year, args.month)
    verify_scaler_stats(df, feature_cols, signals_cfg)
    fit_and_save_preprocessor(df, feature_cols, Path(args.output), args.context_length)


if __name__ == "__main__":
    main()

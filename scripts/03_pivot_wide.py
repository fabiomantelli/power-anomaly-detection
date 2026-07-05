"""
03_pivot_wide.py
Converte o Parquet de formato longo (long) para largo (wide).

Long:  colunas = [timestamp_us, point_id, value, quality]
Wide:  colunas = [timestamp_us, <point_id_1>, <point_id_2>, ...]

Particionamento de saída: OUTPUT_DIR/year=YYYY/month=MM/data.parquet
Compressão: zstd nível 3 (melhor ratio que snappy para wide float32).

Uso:
    python scripts/03_pivot_wide.py \
        --input  D:/openHistorian_export \
        --output D:/export_wide \
        --signals configs/signals.json \
        [--year 2022] [--month 3] [--workers 4]

Sem --year/--month processa todos os anos/meses encontrados.
--workers paraleliza por processo entre partições (year, month), que são
independentes entre si.
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def pivot_partition(
    input_dir: Path,
    output_dir: Path,
    year: int,
    month: int,
    point_id_list: list[int],
) -> None:
    """Pivota todos os dias de um mês e grava um único Parquet wide."""
    out_file = output_dir / f"year={year}" / f"month={month:02d}" / "data.parquet"
    if out_file.exists():
        log.info("Pulando %04d-%02d (já existe)", year, month)
        return

    # Coleta todos os arquivos do mês
    day_files = sorted(
        (input_dir / f"year={year}" / f"month={month:02d}").glob("day=*/data.parquet")
    )
    if not day_files:
        log.warning("Nenhum arquivo encontrado para %04d-%02d", year, month)
        return

    log.info("Pivotando %04d-%02d (%d dias) ...", year, month, len(day_files))

    frames: list[pd.DataFrame] = [
        pd.read_parquet(day_file, columns=["timestamp_us", "point_id", "value"])
        for day_file in day_files
    ]
    long_df = pd.concat(frames, ignore_index=True)

    # Em caso de duplicatas de (timestamp_us, point_id), mantém a primeira —
    # mesma semântica do antigo pivot_table(aggfunc="first"), mas dropar antes
    # de um único unstake do mês é mais barato que N pivot_table (um por dia)
    # seguidos de concat.
    long_df.drop_duplicates(subset=["timestamp_us", "point_id"], keep="first", inplace=True)
    combined = long_df.set_index(["timestamp_us", "point_id"])["value"].unstack("point_id").sort_index()
    combined.columns = [str(c) for c in combined.columns]

    # Garantir que todas as colunas esperadas existam (preencher NaN se PMU offline)
    expected_cols = [str(pid) for pid in point_id_list]
    for col in expected_cols:
        if col not in combined.columns:
            combined[col] = float("nan")
    combined = combined[expected_cols]  # ordem determinística

    combined = combined.reset_index()  # timestamp_us volta a ser coluna
    combined = combined.astype({col: "float32" for col in expected_cols})

    table = pa.Table.from_pandas(combined, preserve_index=False)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_file, compression="zstd", compression_level=3)
    log.info("  Gravado: %s (%d linhas, %d colunas)", out_file, len(combined), len(expected_cols))


def main():
    parser = argparse.ArgumentParser(description="Conversão long → wide Parquet")
    parser.add_argument("--input",   default="D:/openHistorian_export", help="Diretório de entrada (long)")
    parser.add_argument("--output",  default="D:/export_wide",          help="Diretório de saída (wide)")
    parser.add_argument("--signals", default="configs/signals.json",    help="Arquivo de sinais")
    parser.add_argument("--year",    type=int, default=None,            help="Processar apenas este ano")
    parser.add_argument("--month",   type=int, default=None,            help="Processar apenas este mês")
    parser.add_argument("--workers", type=int, default=4,                help="Processos paralelos (partições year/month são independentes)")
    args = parser.parse_args()

    with open(args.signals, encoding="utf-8") as f:
        signals_cfg = json.load(f)
    point_id_list: list[int] = signals_cfg["all_point_ids"]

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    # Descobrir partições disponíveis
    if args.year and args.month:
        partitions = [(args.year, args.month)]
    elif args.year:
        month_dirs = sorted((input_dir / f"year={args.year}").glob("month=*"))
        partitions = [(args.year, int(d.name.split("=")[1])) for d in month_dirs]
    else:
        partitions = []
        for year_dir in sorted(input_dir.glob("year=*")):
            year = int(year_dir.name.split("=")[1])
            for month_dir in sorted(year_dir.glob("month=*")):
                month = int(month_dir.name.split("=")[1])
                partitions.append((year, month))

    log.info("Partições a processar: %d", len(partitions))

    n_workers = max(1, min(args.workers, len(partitions))) if partitions else 1
    if n_workers <= 1:
        for year, month in partitions:
            pivot_partition(input_dir, output_dir, year, month, point_id_list)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(pivot_partition, input_dir, output_dir, year, month, point_id_list)
                for year, month in partitions
            ]
            for future in as_completed(futures):
                future.result()

    log.info("Pivotamento concluído.")


if __name__ == "__main__":
    main()

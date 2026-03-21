"""
02_export_to_parquet.py
Exportação diária do openHistorian 2 → Parquet (formato longo).

Particionamento: OUTPUT_DIR/year=YYYY/month=MM/day=DD/data.parquet
Schema: timestamp_us (int64 UTC µs), point_id (uint64), value (float32), quality (uint32)

Uso:
    python scripts/02_export_to_parquet.py --host 192.168.x.x \
        --start 2022-01-01 --end 2024-01-01 \
        --output D:/openHistorian_export

Suporte a retomada: dias já exportados são pulados automaticamente.
Para paralelizar, usar --start/--end para dividir em semestres e rodar em paralelo.
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from openHistorian.historianConnection import historianConnection
from openHistorian.historianKey import historianKey
from openHistorian.historianValue import historianValue
from snapDB.pointIDMatchFilter import pointIDMatchFilter
from snapDB.timestampSeekFilter import timestampSeekFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SCHEMA = pa.schema([
    pa.field("timestamp_us", pa.int64()),   # microssegundos UTC
    pa.field("point_id",     pa.uint64()),
    pa.field("value",        pa.float32()),
    pa.field("quality",      pa.uint32()),
])


def connect(host: str, instance_name: str) -> tuple:
    """Abre uma conexão nova ao openHistorian. Retorna (historian, instance)."""
    historian = historianConnection(host)
    historian.Connect()
    name = instance_name or historian.InstanceNames[0]
    instance = historian.OpenInstance(name)
    return historian, instance, name


BATCH_SIZE = 2_000_000  # registros por batch (~48 MB em listas Python)


def export_day(host: str, instance_name: str, point_id_list: list[int], date: datetime, out_file: Path) -> int:
    """Abre uma conexão nova, exporta um dia gravando em batches. Retorna número de registros."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(".tmp.parquet")

    historian, instance, _ = connect(host, instance_name)
    total = 0
    try:
        next_day = date + timedelta(days=1)
        time_filter  = timestampSeekFilter.CreateFromRange(date, next_day)
        point_filter = pointIDMatchFilter.CreateFromList(point_id_list)
        reader = instance.Read(time_filter, point_filter)

        key, value = historianKey(), historianValue()
        ts_list:  list[int]   = []
        pid_list: list[int]   = []
        val_list: list[float] = []
        qfl_list: list[int]   = []

        def flush(writer_ref: list) -> None:
            table = pa.table(
                {"timestamp_us": ts_list, "point_id": pid_list,
                 "value": val_list, "quality": qfl_list},
                schema=SCHEMA,
            )
            if not writer_ref:
                writer_ref.append(pq.ParquetWriter(tmp_file, SCHEMA, compression="snappy"))
            writer_ref[0].write_table(table)
            ts_list.clear(); pid_list.clear(); val_list.clear(); qfl_list.clear()

        writer_ref: list = []
        while reader.Read(key, value):
            ts_list.append(int(key.AsDateTime.timestamp() * 1_000_000))
            pid_list.append(int(key.PointID))
            val_list.append(float(value.AsSingle))
            qfl_list.append(int(value.AsQuality.value))
            total += 1
            if len(ts_list) >= BATCH_SIZE:
                flush(writer_ref)

        if ts_list:
            flush(writer_ref)

        if writer_ref:
            writer_ref[0].close()
    finally:
        try:
            instance.Dispose()
            historian.Disconnect()
        except Exception:
            pass

    if total == 0:
        log.warning("Nenhum dado para %s", date.date())
        if tmp_file.exists():
            tmp_file.unlink()
        return 0

    tmp_file.rename(out_file)
    return total


def main():
    parser = argparse.ArgumentParser(description="Exportação openHistorian → Parquet (long format)")
    parser.add_argument("--host",     required=True,                        help="IP do servidor openHistorian")
    parser.add_argument("--start",    default="2022-01-01",                 help="Data inicial (YYYY-MM-DD)")
    parser.add_argument("--end",      default="2024-01-01",                 help="Data final exclusiva (YYYY-MM-DD)")
    parser.add_argument("--output",   default="D:/openHistorian_export",    help="Diretório de saída")
    parser.add_argument("--signals",  default="configs/signals.json",       help="Arquivo de sinais")
    parser.add_argument("--instance", default=None,                         help="Nome da instância openHistorian")
    args = parser.parse_args()

    with open(args.signals, encoding="utf-8") as f:
        signals_cfg = json.load(f)

    point_id_list: list[int] = signals_cfg["all_point_ids"]
    if not point_id_list:
        log.error("all_point_ids vazio em %s — execute 01_list_signals.py primeiro", args.signals)
        raise SystemExit(1)

    # Resolve o nome da instância uma vez antes do loop principal
    log.info("Conectando a %s para resolver instância ...", args.host)
    historian, instance, instance_name = connect(args.host, args.instance)
    log.info("Instância: %s | %d point IDs", instance_name, len(point_id_list))
    try:
        instance.Dispose()
        historian.Disconnect()
    except Exception:
        pass

    output_dir = Path(args.output)
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date   = datetime.strptime(args.end,   "%Y-%m-%d")

    total_records = 0
    current = start_date
    while current < end_date:
        out_file = (
            output_dir
            / f"year={current.year}"
            / f"month={current.month:02d}"
            / f"day={current.day:02d}"
            / "data.parquet"
        )
        if out_file.exists():
            log.info("Pulando %s (já exportado)", current.date())
            current += timedelta(days=1)
            continue

        log.info("Exportando %s ...", current.date())
        n = export_day(args.host, instance_name, point_id_list, current, out_file)
        total_records += n
        log.info("  %s → %d registros", current.date(), n)
        current += timedelta(days=1)

    log.info("Exportação concluída. Total: %d registros", total_records)


if __name__ == "__main__":
    main()

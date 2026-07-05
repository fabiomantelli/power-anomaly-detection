"""
02_export_to_parquet.py
Exportação diária do openHistorian 2 → Parquet (formato longo).

Particionamento: OUTPUT_DIR/year=YYYY/month=MM/day=DD/data.parquet
Schema: timestamp_us (int64 UTC µs), point_id (uint64), value (float32), quality (uint32)

Uso:
    python scripts/02_export_to_parquet.py --host 192.168.x.x \
        --start 2022-01-01 --end 2024-01-01 \
        --output D:/openHistorian_export --workers 4 --grid-hz 60

Suporte a retomada: dias já exportados são pulados automaticamente.
--workers paraleliza por processo (cada worker cuida de um intervalo contíguo de
dias com sua própria conexão ao openHistorian). Calibre o valor observando
CPU/rede locais e a carga no servidor antes de rodar o backfill completo.

--grid-hz {50,60} filtra point_id_list pela frequência nominal da rede elétrica
(60 Hz = só Brasil; 50 Hz = demais países — ver by_grid_hz em configs/signals.json,
gerado por 01_list_signals.py). Sem --grid-hz, exporta tudo junto como antes.
Quando usado, a saída fica em OUTPUT_DIR/hz=50|60/year=.../month=.../day=.../data.parquet
— uma subpasta por frequência, para nunca colidir com uma exportação da outra
frequência nem com uma exportação antiga sem --grid-hz.
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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

# Ticks .NET (unidades de 100ns) entre 0001-01-01 (Empty.DATETIME do oh_gsf) e
# 1970-01-01 (epoch Unix). Usado para converter key.Timestamp -> microssegundos
# UTC com aritmética inteira vetorizada, sem passar por datetime por registro.
UNIX_EPOCH_TICKS = 621_355_968_000_000_000


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
        # Acumula campos crus (sem construir datetime/enum por registro) e só
        # converte em lote no flush, com numpy vetorizado.
        ts_ticks:  list[int] = []
        pid_list:  list[int] = []
        val_bits:  list[int] = []
        qfl_bits:  list[int] = []

        def flush(writer_ref: list) -> None:
            ts_us  = (np.asarray(ts_ticks, dtype=np.uint64) - UNIX_EPOCH_TICKS) // 10
            pid_np = np.asarray(pid_list, dtype=np.uint64)
            val_np = np.asarray(val_bits, dtype=np.uint64).astype(np.uint32).view(np.float32)
            qfl_np = np.asarray(qfl_bits, dtype=np.uint64).astype(np.uint32)

            table = pa.table(
                {"timestamp_us": ts_us.astype(np.int64), "point_id": pid_np,
                 "value": val_np, "quality": qfl_np},
                schema=SCHEMA,
            )
            if not writer_ref:
                writer_ref.append(pq.ParquetWriter(tmp_file, SCHEMA, compression="snappy"))
            writer_ref[0].write_table(table)
            ts_ticks.clear(); pid_list.clear(); val_bits.clear(); qfl_bits.clear()

        writer_ref: list = []
        while reader.Read(key, value):
            ts_ticks.append(int(key.Timestamp))
            pid_list.append(int(key.PointID))
            val_bits.append(int(value.Value1))
            qfl_bits.append(int(value.Value3))
            total += 1
            if len(ts_ticks) >= BATCH_SIZE:
                flush(writer_ref)

        if ts_ticks:
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


def day_output_path(output_dir: Path, date: datetime, grid_hz: int | None) -> Path:
    """Caminho do arquivo do dia. Namespaced por hz=50|60 quando --grid-hz é usado,
    para nunca colidir com uma exportação da outra frequência (ou sem filtro)."""
    root = output_dir / f"hz={grid_hz}" if grid_hz is not None else output_dir
    return (
        root
        / f"year={date.year}"
        / f"month={date.month:02d}"
        / f"day={date.day:02d}"
        / "data.parquet"
    )


def export_date_chunk(
    host: str,
    instance_name: str,
    point_id_list: list[int],
    output_dir_str: str,
    date_isos: list[str],
    grid_hz: int | None = None,
) -> int:
    """Executado em um processo worker: exporta uma lista de dias, sequencialmente.

    Cada worker abre suas próprias conexões (uma por dia, como já era) e cuida de um
    subconjunto de dias disjunto dos demais workers, então nunca há dois processos
    escrevendo o mesmo arquivo `day=DD/data.parquet`.
    """
    output_dir = Path(output_dir_str)
    total = 0
    for date_iso in date_isos:
        date = datetime.strptime(date_iso, "%Y-%m-%d")
        out_file = day_output_path(output_dir, date, grid_hz)
        if out_file.exists():
            log.info("Pulando %s (já exportado)", date.date())
            continue

        log.info("Exportando %s ...", date.date())
        n = export_day(host, instance_name, point_id_list, date, out_file)
        total += n
        log.info("  %s → %d registros", date.date(), n)

    return total


def chunkify(items: list, n: int) -> list[list]:
    """Divide `items` em `n` pedaços contíguos de tamanho o mais igual possível."""
    n = max(1, min(n, len(items))) if items else 1
    quotient, remainder = divmod(len(items), n)
    chunks: list[list] = []
    start = 0
    for i in range(n):
        size = quotient + (1 if i < remainder else 0)
        if size:
            chunks.append(items[start:start + size])
            start += size
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Exportação openHistorian → Parquet (long format)")
    parser.add_argument("--host",     required=True,                        help="IP do servidor openHistorian")
    parser.add_argument("--start",    default="2022-01-01",                 help="Data inicial (YYYY-MM-DD)")
    parser.add_argument("--end",      default="2024-01-01",                 help="Data final exclusiva (YYYY-MM-DD)")
    parser.add_argument("--output",   default="D:/openHistorian_export",    help="Diretório de saída")
    parser.add_argument("--signals",  default="configs/signals.json",       help="Arquivo de sinais")
    parser.add_argument("--instance", default=None,                         help="Nome da instância openHistorian")
    parser.add_argument("--workers",  type=int, default=4,                  help="Processos paralelos (calibre contra o servidor antes do backfill completo)")
    parser.add_argument("--grid-hz",  type=int, default=None, choices=[50, 60], help="Filtra por frequência da rede (60=Brasil, 50=demais países). Sem isso, exporta tudo.")
    args = parser.parse_args()

    with open(args.signals, encoding="utf-8") as f:
        signals_cfg = json.load(f)

    if args.grid_hz is not None:
        if "by_grid_hz" not in signals_cfg:
            log.error(
                "configs/signals.json não tem 'by_grid_hz' — re-execute 01_list_signals.py "
                "(versão atualizada) antes de usar --grid-hz."
            )
            raise SystemExit(1)
        point_id_list: list[int] = signals_cfg["by_grid_hz"].get(str(args.grid_hz), [])
    else:
        point_id_list = signals_cfg["all_point_ids"]

    if not point_id_list:
        log.error("Nenhum point ID encontrado (signals=%s, grid_hz=%s) — execute 01_list_signals.py primeiro", args.signals, args.grid_hz)
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

    all_dates: list[str] = []
    current = start_date
    while current < end_date:
        all_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    if not all_dates:
        log.warning("Intervalo --start/--end vazio, nada a exportar.")
        return

    chunks = chunkify(all_dates, args.workers)
    log.info("Exportando %d dias com %d worker(s) ...", len(all_dates), len(chunks))

    total_records = 0
    if len(chunks) == 1:
        total_records = export_date_chunk(args.host, instance_name, point_id_list, str(output_dir), chunks[0], args.grid_hz)
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [
                pool.submit(export_date_chunk, args.host, instance_name, point_id_list, str(output_dir), chunk, args.grid_hz)
                for chunk in chunks
            ]
            for future in as_completed(futures):
                total_records += future.result()

    log.info("Exportação concluída. Total: %d registros", total_records)


if __name__ == "__main__":
    main()

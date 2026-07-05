"""
01_list_signals.py
Discovery de sinais disponíveis no openHistorian 2.
Salva o mapeamento point_id → sinal/PMU/país/frequência em configs/signals.json.

Uso:
    python scripts/01_list_signals.py --host 192.168.x.x [--instance PPA]

Cada point ID recebe um "grid_hz" (50 ou 60) derivado do país do device (só o
Brasil é 60 Hz). O resultado inclui "by_grid_hz" para uso por
02_export_to_parquet.py / 03_pivot_wide.py com a flag --grid-hz.
"""

import argparse
import json
from pathlib import Path

from openHistorian.historianConnection import historianConnection
from openHistorian.measurementRecord import SignalType


SIGNAL_TYPES = [SignalType.FREQ, SignalType.DFDT, SignalType.VPHM, SignalType.VPHA]

# Mapeamento de acronimos de dispositivo → país.
# ATENÇÃO: a rede cresce com o tempo (novas PMUs entram no historian sem aviso
# prévio a este script). Qualquer device não listado aqui cai em "Unknown" no
# by_country/point_id_to_meta — rode 01_list_signals.py periodicamente e
# confira o resumo impresso no final para pegar instituições novas.
DEVICE_COUNTRY: dict[str, str] = {
    # Chile
    "UDA":    "Chile",
    "UTEM":   "Chile",
    "UTA":    "Chile",
    "USACH":  "Chile",
    "UTALCA": "Chile",
    "UDEC":   "Chile",
    "UFRO":   "Chile",
    # Argentina
    "UNT":    "Argentina",
    "UNSJ":   "Argentina",
    "UNLP":   "Argentina",
    "UNCO":   "Argentina",
    # Uruguai
    "UTEC":   "Uruguay",
    # Portugal
    "INESCTEC": "Portugal",
    "ISE":      "Portugal",
    # Itália
    "POLIMI": "Italy",
    # Suíça
    "ZHAW":   "Switzerland",
    # Brasil
    "COPPE":   "Brazil",
    "PTI":     "Brazil",
    "UFAC":    "Brazil",
    "UFAM":    "Brazil",
    "UFBA":    "Brazil",
    "UFC":     "Brazil",
    "UFES":    "Brazil",
    "UFJF":    "Brazil",
    "UFMA":    "Brazil",
    "UFMG":    "Brazil",
    "UFMS":    "Brazil",
    "UFMT":    "Brazil",
    "UFPA":    "Brazil",
    "UFPE":    "Brazil",
    "UFPI":    "Brazil",
    "UFPR":    "Brazil",
    "UFRGS":   "Brazil",
    "UFRR":    "Brazil",
    "UFSC":    "Brazil",
    "UFT":     "Brazil",
    "UNB":     "Brazil",
    "UNICAMP": "Brazil",
    "UNIFAP":  "Brazil",
    "UNIFEI":  "Brazil",
    "UNIPAMPA": "Brazil",
    "UNIR":    "Brazil",
    "USP-SC":  "Brazil",
    "UTFPR":   "Brazil",
    "POCKET-PMU_UFSC":    "Brazil",
    "POCKET-PMU_UNICAMP": "Brazil",
    "POCKET-PMU_QUINTAO": "Brazil",
    "PMU_TESTE_51":       "Brazil",
}

# Frequência nominal da rede elétrica: só o Brasil opera em 60 Hz, os demais
# países do dataset (Chile, Argentina, Uruguai, Portugal, Itália, Suíça) são
# 50 Hz. Usado por 02_export_to_parquet.py / 03_pivot_wide.py (--grid-hz) para
# separar as duas taxas de amostragem, que não fazem sentido misturadas no
# mesmo dataset wide.
BRAZIL_GRID_HZ = 60
DEFAULT_GRID_HZ = 50


def grid_hz_for_country(country: str) -> int:
    return BRAZIL_GRID_HZ if country == "Brazil" else DEFAULT_GRID_HZ


def discover_signals(host: str, instance_name: str | None = None) -> dict:
    historian = historianConnection(host)
    historian.Connect()
    historian.RefreshMetadata()
    metadata = historian.Metadata

    if instance_name is None:
        instance_name = historian.InstanceNames[0]
    print(f"Instância openHistorian: {instance_name}")

    all_point_ids: list[int] = []
    # Sem chaves pré-definidas: qualquer país (inclusive "Unknown") aparece no
    # resultado, em vez de ser descartado silenciosamente por não estar numa
    # lista fixa de países esperados.
    by_country: dict[str, list[int]] = {}
    by_grid_hz: dict[int, list[int]] = {}
    by_signal_type: dict[str, list[int]] = {st.name: [] for st in SIGNAL_TYPES}
    point_id_to_meta: dict[str, dict] = {}
    pmus_seen: set[str] = set()

    for sig_type in SIGNAL_TYPES:
        records = metadata.GetMeasurementsBySignalType(sig_type, instance_name)
        for r in records:
            pid = int(r.pointID)
            device = r.deviceAcronym or ""
            country = DEVICE_COUNTRY.get(device, "Unknown")
            grid_hz = grid_hz_for_country(country)

            all_point_ids.append(pid)
            by_signal_type[sig_type.name].append(pid)
            by_country.setdefault(country, []).append(pid)
            by_grid_hz.setdefault(grid_hz, []).append(pid)

            point_id_to_meta[str(pid)] = {
                "device": device,
                "signal_type": sig_type.name,
                "signal_reference": r.signalReference,
                "description": r.description,
                "country": country,
                "grid_hz": grid_hz,
                "tag_name": r.pointTag,
            }
            pmus_seen.add(device)

            print(f"  {pid:>8} | {device:<15} | {sig_type.name:<6} | {r.description}")

    historian.Disconnect()

    return {
        "host": host,
        "instance": instance_name,
        "signal_types": [st.name for st in SIGNAL_TYPES],
        "channels_per_pmu": ["va_mag", "vb_mag", "vc_mag", "va_ang", "vb_ang", "vc_ang", "freq", "dfreq"],
        "pmus": sorted(pmus_seen),
        "all_point_ids": sorted(set(all_point_ids)),
        "by_country": {k: sorted(set(v)) for k, v in by_country.items()},
        "by_grid_hz": {str(k): sorted(set(v)) for k, v in by_grid_hz.items()},
        "by_signal_type": {k: sorted(v) for k, v in by_signal_type.items()},
        "point_id_to_meta": point_id_to_meta,
    }


def main():
    parser = argparse.ArgumentParser(description="Discovery de sinais no openHistorian 2")
    parser.add_argument("--host", required=True, help="IP do servidor openHistorian (via VPN)")
    parser.add_argument("--instance", default=None, help="Nome da instância (ex: PPA)")
    parser.add_argument("--output", default="configs/signals.json", help="Arquivo de saída")
    args = parser.parse_args()

    signals = discover_signals(args.host, args.instance)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2, ensure_ascii=False)

    n_pmus = len(signals["pmus"])
    n_pids = len(signals["all_point_ids"])
    print(f"\nSalvo em {out_path}")
    print(f"  PMUs encontradas : {n_pmus}")
    print(f"  Point IDs totais : {n_pids}")
    for country, pids in signals["by_country"].items():
        print(f"  {country:<12}: {len(pids)} sinais")
    for grid_hz, pids in signals["by_grid_hz"].items():
        print(f"  {grid_hz} Hz      : {len(pids)} sinais")
    for sig_type, pids in signals["by_signal_type"].items():
        print(f"  {sig_type:<8}: {len(pids)} sinais")


if __name__ == "__main__":
    main()

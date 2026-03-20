"""
01_list_signals.py
Discovery de sinais disponíveis no openHistorian 2.
Salva o mapeamento point_id → sinal/PMU/país em configs/signals.json.

Uso:
    python scripts/01_list_signals.py --host 192.168.x.x [--instance PPA]
"""

import argparse
import json
from pathlib import Path

from openHistorian.historianConnection import historianConnection
from openHistorian.measurementRecord import SignalType


SIGNAL_TYPES = [SignalType.FREQ, SignalType.DFDT, SignalType.VPHM, SignalType.VPHA]

# Mapeamento de acronimos de dispositivo → país
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
}


def discover_signals(host: str, instance_name: str | None = None) -> dict:
    historian = historianConnection(host)
    historian.Connect()
    historian.RefreshMetadata()
    metadata = historian.Metadata

    if instance_name is None:
        instance_name = historian.InstanceNames[0]
    print(f"Instância openHistorian: {instance_name}")

    all_point_ids: list[int] = []
    by_country: dict[str, list[int]] = {"Chile": [], "Argentina": []}
    by_signal_type: dict[str, list[int]] = {st.name: [] for st in SIGNAL_TYPES}
    point_id_to_meta: dict[str, dict] = {}
    pmus_seen: set[str] = set()

    for sig_type in SIGNAL_TYPES:
        records = metadata.GetMeasurementsBySignalType(sig_type, instance_name)
        for r in records:
            pid = int(r.pointID)
            device = r.deviceAcronym or ""
            country = DEVICE_COUNTRY.get(device, "Unknown")

            all_point_ids.append(pid)
            by_signal_type[sig_type.name].append(pid)
            if country in by_country:
                by_country[country].append(pid)

            point_id_to_meta[str(pid)] = {
                "device": device,
                "signal_type": sig_type.name,
                "signal_reference": r.signalReference,
                "description": r.description,
                "country": country,
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
    for sig_type, pids in signals["by_signal_type"].items():
        print(f"  {sig_type:<8}: {len(pids)} sinais")


if __name__ == "__main__":
    main()

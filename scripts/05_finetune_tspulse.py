"""
05_finetune_tspulse.py
Fine-tuning do TSPulse (IBM Granite) em dados normais de PMU.

O modelo aprende a reconstruir séries temporais normais.
Anomalias são detectadas como alta perda de reconstrução em inferência.

Uso (na instância vast.ai com H100):
    python scripts/05_finetune_tspulse.py \
        --data-glob "/workspace/pmu_data/year=2022/month=*/data.parquet" \
        --signals   configs/signals.json \
        --preprocessor checkpoints/pmu_preprocessor \
        --output    checkpoints/tspulse_pmu_finetuned \
        [--epochs 10] [--batch-size 64] [--lr 5e-5]
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

CONTEXT_LENGTH = 512   # 512 amostras ≈ 8.5s a 60 Hz
STRIDE         = 60    # 1s de passo entre janelas


def build_data_loader(
    data_glob: str,
    feature_cols: list[str],
    preprocessor,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from pmu_dataset import PMUSlidingWindowDataset

    dataset = PMUSlidingWindowDataset(
        parquet_glob=data_glob,
        feature_cols=feature_cols,
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
        preprocessor=preprocessor,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning do TSPulse em dados normais de PMU")
    parser.add_argument("--data-glob",    required=True,                              help="Glob dos Parquet wide")
    parser.add_argument("--signals",      default="configs/signals.json",             help="Arquivo de sinais")
    parser.add_argument("--preprocessor", default="checkpoints/pmu_preprocessor",    help="Preprocessor salvo")
    parser.add_argument("--output",       default="checkpoints/tspulse_pmu_finetuned", help="Diretório de saída")
    parser.add_argument("--epochs",       type=int,   default=10,                     help="Épocas de treinamento")
    parser.add_argument("--batch-size",   type=int,   default=64,                     help="Batch size")
    parser.add_argument("--lr",           type=float, default=5e-5,                   help="Learning rate")
    parser.add_argument("--num-workers",  type=int,   default=4,                      help="Workers do DataLoader")
    parser.add_argument("--grad-accum",   type=int,   default=4,                      help="Gradient accumulation steps")
    args = parser.parse_args()

    # Importações que requerem GPU/ambiente vast.ai
    try:
        from accelerate import Accelerator
        from tsfm_public import TimeSeriesPreprocessor
        from tsfm_public.models.tspulse import TSPulseForReconstruction
    except ImportError as e:
        log.error("Dependência ausente: %s. Instale granite-tsfm e accelerate.", e)
        raise

    with open(args.signals, encoding="utf-8") as f:
        signals_cfg = json.load(f)
    feature_cols = [str(pid) for pid in signals_cfg["all_point_ids"]]
    n_channels   = len(feature_cols)
    log.info("Canais: %d", n_channels)

    # Carrega preprocessor para normalização online
    preprocessor = TimeSeriesPreprocessor.from_pretrained(args.preprocessor)

    # Configura Accelerator (BF16, gradient accumulation)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum,
        log_with="tensorboard",
        project_dir=args.output,
    )

    # Modelo TSPulse
    model = TSPulseForReconstruction.from_pretrained(
        "ibm-granite/granite-timeseries-tspulse-r1",
        num_input_channels=n_channels,
        context_length=CONTEXT_LENGTH,
        mask_type="user",
    )
    log.info(
        "Parâmetros treináveis: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # DataLoader
    train_loader = build_data_loader(
        data_glob=args.data_glob,
        feature_cols=feature_cols,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Otimizador + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Estimativa do número de passos (conservadora — streaming não tem len exato)
    estimated_steps = 50_000 * args.epochs  # ajustar após 1ª época
    scheduler = CosineAnnealingLR(optimizer, T_max=estimated_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    accelerator.init_trackers("tspulse_pmu_finetuning")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss    = outputs.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches  += 1
            global_step += 1

            if global_step % 500 == 0:
                avg = epoch_loss / n_batches
                accelerator.log({"train/loss": avg, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
                log.info("Epoch %d | step %d | loss=%.6f", epoch, global_step, avg)

        avg_loss = epoch_loss / max(n_batches, 1)
        log.info("Epoch %d concluída | loss_média=%.6f", epoch, avg_loss)

        # Checkpoint por época
        ckpt_dir = Path(args.output) / f"epoch_{epoch:02d}"
        accelerator.save_state(str(ckpt_dir))
        log.info("Checkpoint salvo em %s", ckpt_dir)

    # Salva modelo final
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(str(output_dir / "model"))
    log.info("Modelo final salvo em %s/model", output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()

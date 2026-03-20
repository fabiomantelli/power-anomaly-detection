"""
pmu_dataset.py
Dataset de janelas deslizantes para dados de PMU (Parquet wide).

PMUSlidingWindowDataset: IterableDataset que lê arquivos Parquet em streaming,
mantém um buffer e emite janelas de comprimento fixo com stride configurável.

Exemplo de uso:
    dataset = PMUSlidingWindowDataset(
        parquet_glob="D:/export_wide/year=2022/month=*/data.parquet",
        feature_cols=feature_cols,
        context_length=512,
        stride=60,
    )
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    for batch in loader:
        # batch["past_values"]: (B, 512, N_channels)
        ...
"""

import glob as _glob
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)


class PMUSlidingWindowDataset(IterableDataset):
    """
    Dataset iterável de janelas deslizantes sobre arquivos Parquet wide de PMU.

    Args:
        parquet_glob: Padrão glob para os arquivos Parquet wide.
            Ex: "D:/export_wide/year=2022/month=*/data.parquet"
        feature_cols: Lista de nomes de colunas (point_ids como strings).
        context_length: Número de amostras por janela (default 512 = ~8.5s a 60 Hz).
        stride: Passo entre janelas consecutivas (default 60 = 1s a 60 Hz).
        preprocessor: Instância opcional de TimeSeriesPreprocessor para normalização.
            Se None, os valores são usados sem escalonamento.
        batch_size_parquet: Tamanho do lote ao ler cada arquivo Parquet (linhas).
        drop_nan_windows: Se True, descarta janelas com qualquer NaN.
    """

    def __init__(
        self,
        parquet_glob: str,
        feature_cols: list[str],
        context_length: int = 512,
        stride: int = 60,
        preprocessor=None,
        batch_size_parquet: int = 36_000,
        drop_nan_windows: bool = True,
    ):
        self.files              = sorted(_glob.glob(parquet_glob))
        self.feature_cols       = feature_cols
        self.context_len        = context_length
        self.stride             = stride
        self.preprocessor       = preprocessor
        self.batch_size_parquet = batch_size_parquet
        self.drop_nan_windows   = drop_nan_windows

        if not self.files:
            raise FileNotFoundError(f"Nenhum arquivo encontrado para o padrão: {parquet_glob}")
        log.info("PMUSlidingWindowDataset: %d arquivos, %d canais", len(self.files), len(feature_cols))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer = np.empty((0, len(self.feature_cols)), dtype=np.float32)

        for path in self.files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                batch_size=self.batch_size_parquet,
                columns=self.feature_cols,
            ):
                arr = batch.to_pandas().values.astype(np.float32)

                if self.preprocessor is not None:
                    # Normalização por canal (Z-score); preprocessor.transform espera DataFrame
                    import pandas as pd
                    df_batch = pd.DataFrame(arr, columns=self.feature_cols)
                    arr = self.preprocessor.preprocess(df_batch)[self.feature_cols].values.astype(np.float32)

                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                buffer = np.concatenate([buffer, arr], axis=0)

                # Emite janelas completas
                n_windows = (len(buffer) - self.context_len) // self.stride
                for i in range(n_windows):
                    start = i * self.stride
                    window = buffer[start : start + self.context_len]  # (context_len, n_channels)

                    if self.drop_nan_windows and np.isnan(window).any():
                        continue

                    yield {"past_values": torch.tensor(window, dtype=torch.float32)}

                # Mantém apenas o trecho necessário para a próxima janela
                keep = len(buffer) - n_windows * self.stride
                buffer = buffer[-keep:] if keep > 0 else np.empty(
                    (0, len(self.feature_cols)), dtype=np.float32
                )

    def count_windows_estimate(self) -> int:
        """Estimativa do número total de janelas (sem ler os dados)."""
        total_rows = sum(
            pq.read_metadata(f).num_rows for f in self.files
        )
        return max(0, (total_rows - self.context_len) // self.stride)

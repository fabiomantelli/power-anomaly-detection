# AI para Detecção de Anomalias em Sincrofasores 50 Hz

Detecção não supervisionada de anomalias em dados de PMU (Phase Measurement Units) das redes de 50 Hz do **Chile** e da **Argentina**, usando o modelo [IBM TSPulse](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1).

---

## Visão geral

| Item | Detalhe |
|---|---|
| Fonte de dados | openHistorian 2 (servidor remoto via VPN) |
| Taxa de amostragem | 60 amostras/segundo por PMU |
| PMUs | 5–20 unidades (Chile + Argentina) |
| Canais por PMU | 8 (Va/Vb/Vc magnitude e ângulo, freq, ROCOF) |
| Período | 2 anos (2022–2023) |
| Modelo | TSPulse — 1 M parâmetros, zero-shot, domínio dual temporal+espectral |
| Hardware de treinamento | GPU alugada no vast.ai (H100 80 GB) |

---

## Pré-requisitos

### Ambiente local (exportação)

```bash
pip install openhistorian pyarrow>=17 pandas numpy tqdm
```

### Ambiente cloud — vast.ai (treinamento e inferência)

Imagem Docker recomendada: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`

```bash
conda create -n pmu-ai python=3.11 && conda activate pmu-ai
pip install -r requirements.txt
pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.3.1"
```

---

## O que precisa ser feito

### Fase 1 — Configuração inicial (fazer uma vez)

#### 1. Mapear os sinais disponíveis no openHistorian

Conectar ao servidor via VPN e rodar o script de discovery:

```bash
python scripts/01_list_signals.py --host 192.168.x.x
```

Isso cria/atualiza `configs/signals.json` com todos os point IDs das PMUs.

**Ação manual necessária:** abrir `scripts/01_list_signals.py` e preencher o dicionário `DEVICE_COUNTRY` com os acrônimos reais de cada PMU e o país correspondente (Chile ou Argentina). Exemplo:

```python
DEVICE_COUNTRY = {
    "UCHILE01": "Chile",
    "USACH01":  "Chile",
    "UBA01":    "Argentina",
}
```

---

### Fase 2 — Exportação de dados (rodar localmente com VPN ativa)

#### 2. Exportar dados do openHistorian para Parquet

Formato de saída: `D:/openHistorian_export/year=YYYY/month=MM/day=DD/data.parquet`

```bash
# Testar com 1 dia antes de rodar o período completo:
python scripts/02_export_to_parquet.py \
    --host  192.168.x.x \
    --start 2022-01-01 \
    --end   2022-01-02 \
    --output D:/openHistorian_export

# Exportação completa (pode ser dividida em semestres e rodar em paralelo):
python scripts/02_export_to_parquet.py \
    --host  192.168.x.x \
    --start 2022-01-01 \
    --end   2024-01-01 \
    --output D:/openHistorian_export
```

> O script retoma automaticamente a partir do último dia exportado em caso de falha.
>
> Volume estimado (10 PMUs × 8 canais, 2 anos): **15–25 GB** em Parquet Snappy.
> Tempo estimado: 8–15 horas. Para acelerar, rodar dois processos em paralelo com semestres distintos.

**Verificação:**

```python
import pyarrow.dataset as ds
d = ds.dataset("D:/openHistorian_export", format="parquet")
print(d.count_rows())   # deve ser ~302 bilhões para 10 PMUs / 2 anos
print(d.schema)
```

#### 3. Converter de formato longo para largo (wide)

Agrupa os point IDs como colunas. Saída: `D:/export_wide/year=YYYY/month=MM/data.parquet`

```bash
python scripts/03_pivot_wide.py \
    --input  D:/openHistorian_export \
    --output D:/export_wide
```

> Para processar apenas um mês específico: `--year 2022 --month 3`

**Verificação:**

```python
import pandas as pd
df = pd.read_parquet("D:/export_wide/year=2022/month=01/data.parquet")
print(df.shape)    # (N_timestamps, N_channels + 1)
print(df.columns)  # timestamp_us + um ponto ID por coluna
```

---

### Fase 3 — Treinamento (rodar na instância vast.ai)

#### 4. Transferir dados para o vast.ai

No painel do [vast.ai](https://vast.ai), provisionar uma instância com H100 80 GB e obter IP/porta SSH.

```bash
rsync -avz --progress -e "ssh -p <VAST_PORT>" \
      D:/export_wide/ root@<VAST_IP>:/workspace/pmu_data/
```

> Para volumes > 100 GB, considerar vast.ai Persistent Storage.

#### 5. Ajustar o preprocessor (Z-score por canal)

Usar um mês de operação normal (sem eventos conhecidos) para estimar média e desvio padrão de cada canal:

```bash
python scripts/04_fit_preprocessor.py \
    --data-dir /workspace/pmu_data \
    --year 2022 --month 3 \
    --output   checkpoints/pmu_preprocessor
```

**Verificação esperada nos logs:**
- Canal `FREQ`: média ≈ 50.0 Hz, desvio padrão ≈ 0.05–0.2 Hz
- Canais `VPHM`: média ≈ 1.0 pu, desvio padrão ≈ 0.01–0.05 pu

#### 6. Fine-tuning do TSPulse em dados normais

O modelo aprende a reconstruir séries normais. Anomalias = alta perda de reconstrução.

```bash
python scripts/05_finetune_tspulse.py \
    --data-glob    "/workspace/pmu_data/year=2022/month=*/data.parquet" \
    --signals      configs/signals.json \
    --preprocessor checkpoints/pmu_preprocessor \
    --output       checkpoints/tspulse_pmu_finetuned \
    --epochs       10 \
    --batch-size   64
```

> Checkpoints são salvos por época em `checkpoints/tspulse_pmu_finetuned/epoch_XX/`.
> Monitorar loss com TensorBoard: `tensorboard --logdir checkpoints/tspulse_pmu_finetuned`

**Verificação:** a loss de reconstrução deve decrescer a cada época.

---

### Fase 4 — Detecção de anomalias (inferência)

#### 7. Rodar o pipeline de scoring

```bash
# Zero-shot (sem fine-tuning, para validação inicial):
python scripts/06_anomaly_detection.py \
    --data-glob "/workspace/pmu_data/year=2023/month=01/data.parquet" \
    --model     ibm-granite/granite-timeseries-tspulse-r1 \
    --preprocessor checkpoints/pmu_preprocessor \
    --output    results/scores_zeroshot_2023_01.parquet

# Com fine-tuning:
python scripts/06_anomaly_detection.py \
    --data-glob "/workspace/pmu_data/year=2023/month=*/data.parquet" \
    --model     checkpoints/tspulse_pmu_finetuned/model \
    --preprocessor checkpoints/pmu_preprocessor \
    --output    results/scores_finetuned_2023.parquet \
    --threshold 3.0
```

O arquivo de saída contém `timestamp_us`, `anomaly_score` e `anomaly_flag` (1 = anomalia detectada).

**Verificação:** em operação normal, `anomaly_flag` deve ser próximo de 0%. Eventos de rede (quedas de frequência, desequilíbrio de tensão) devem produzir picos claros no `anomaly_score`.

---

## Estrutura de arquivos

```
ai-detection-50hz/
  configs/
    signals.json              # Mapeamento point_id → sinal/PMU/país (gerado por 01)
  scripts/
    01_list_signals.py        # Discovery de sinais no openHistorian
    02_export_to_parquet.py   # Exportação diária (long format)
    03_pivot_wide.py          # Conversão long → wide por mês
    04_fit_preprocessor.py    # Ajuste do Z-score por canal
    05_finetune_tspulse.py    # Fine-tuning do TSPulse
    06_anomaly_detection.py   # Inferência e scoring de anomalias
  src/
    pmu_dataset.py            # PMUSlidingWindowDataset (streaming, IterableDataset)
  checkpoints/
    pmu_preprocessor/         # Escaladores por canal (gerado por 04)
    tspulse_pmu_finetuned/    # Modelo fine-tuned + checkpoints por época (gerado por 05)
  results/                    # Parquet de scores de anomalia (gerado por 06)
  requirements.txt
```

---

## Canais PMU

| Canal | Sinal | Tipo openHistorian |
|---|---|---|
| `va_mag` | Tensão fase A — magnitude (pu) | `VPHM` |
| `vb_mag` | Tensão fase B — magnitude (pu) | `VPHM` |
| `vc_mag` | Tensão fase C — magnitude (pu) | `VPHM` |
| `va_ang` | Tensão fase A — ângulo (graus) | `VPHA` |
| `vb_ang` | Tensão fase B — ângulo (graus) | `VPHA` |
| `vc_ang` | Tensão fase C — ângulo (graus) | `VPHA` |
| `freq`   | Frequência do sistema (Hz) | `FREQ` |
| `dfreq`  | ROCOF — df/dt (Hz/s) | `ROCOF` |

---

## Decisões arquiteturais

| Decisão | Escolha | Motivo |
|---|---|---|
| Formato de dados | Parquet particionado (year/month/day) | Streaming eficiente, columnar para ML |
| Compressão exportação | Snappy | Velocidade de escrita na exportação |
| Compressão wide | Zstd nível 3 | Melhor ratio para floats correlacionados |
| Janela de contexto | 512 amostras (~8.5s a 60 Hz) | Padrão TSPulse, cobre transitórios de rede |
| Stride | 60 amostras (1s) | Resolução de 1s para alertas operacionais |
| Modelo | TSPulse (IBM Granite) | #1 benchmark TSB-AD, zero-shot, 1 M params |
| Estratégia | Fine-tuning em dados normais | Sem rótulos — aprende distribuição normal |
| Hardware cloud | vast.ai H100 80 GB | Custo baixo vs. AWS/GCP |

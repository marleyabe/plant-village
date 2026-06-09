# Plant Village

Classificacao de doencas em folhas de plantas usando TensorFlow, PlantVillage e
transfer learning com MobileNetV2.

O repositorio contem o codigo de treino, avaliacao, predicao, testes automatizados,
metricas do experimento final e o modelo treinado.

## Resultado Principal

Experimento final com o dataset PlantVillage completo:

- Dataset: 54.303 imagens
- Classes: 38
- Divisao: 70% treino, 15% validacao, 15% teste
- Modelo: MobileNetV2 com transfer learning
- Imagem de entrada: 224x224
- Epocas: 20
- Batch size: 8
- Acuracia no teste: 0.953965
- Macro F1 no teste: 0.944077
- Weighted F1 no teste: 0.953609
- Exemplos de teste: 8.146

Metricas completas:

- `artifacts/reports/mobilenet_full_metrics.json`
- `artifacts/reports/mobilenet_full_classification_report.csv`

Modelo treinado:

- `artifacts/models/mobilenet_full.keras`

## Estrutura

```text
.
├── artifacts/
│   ├── models/      # modelo treinado versionado
│   ├── plots/       # graficos e matriz de confusao
│   └── reports/     # metricas, historico e exemplos
├── src/plant_village/
│   ├── config.py    # configuracao do experimento
│   ├── data.py      # carga, split e pipeline tf.data
│   ├── models.py    # baseline CNN, MobileNetV2 e loader do modelo
│   ├── train.py     # treino
│   ├── evaluate.py  # avaliacao
│   ├── predict.py   # predicao em imagem individual
│   └── plots.py     # visualizacoes
├── tests/           # testes das regras de negocio e operacionais
├── TODO_PROJETO.md
├── pyproject.toml
└── uv.lock
```

## Ambiente Local

O projeto usa `uv`.

```bash
uv sync
```

Rodar verificacoes:

```bash
uv run ruff check .
uv run pytest
```

Abrir Jupyter, se necessario:

```bash
uv run jupyter lab
```

## Dataset

O dataset usado e o PlantVillage disponivel no TensorFlow Datasets:

https://tensorflow.google.cn/datasets/catalog/plant_village

O download e preparo sao feitos automaticamente pelo TensorFlow Datasets no primeiro treino
ou avaliacao. Os dados ficam em `data/tfds/`, que nao e versionado no Git porque e grande e
reprodutivel.

## Treino

Treino principal com MobileNetV2:

```bash
uv run plant-train --model mobilenet --run-name mobilenet_full --epochs 20 --batch-size 8
```

Treino da CNN baseline:

```bash
uv run plant-train --model baseline --run-name baseline_full --epochs 20
```

Modo rapido para validar codigo sem esperar o dataset completo:

```bash
uv run plant-train --model baseline --run-name baseline_debug --epochs 2 --max-examples 1000
```

## Avaliacao

Avaliar o modelo treinado:

```bash
uv run plant-evaluate artifacts/models/mobilenet_full.keras
```

Quando existe `artifacts/reports/mobilenet_full_metadata.json`, a avaliacao reutiliza
automaticamente `seed`, `image_size`, `batch_size` e `max_examples` do treino. Isso evita
avaliar o modelo com uma configuracao diferente da usada no experimento.

Saidas geradas:

- `artifacts/reports/mobilenet_full_metrics.json`
- `artifacts/reports/mobilenet_full_classification_report.csv`
- `artifacts/reports/mobilenet_full_prediction_examples.csv`
- `artifacts/plots/mobilenet_full_confusion_matrix.png`
- `artifacts/plots/mobilenet_full_correct_examples.png`
- `artifacts/plots/mobilenet_full_error_examples.png`

## Predicao

Classificar uma imagem individual:

```bash
uv run plant-predict artifacts/models/mobilenet_full.keras caminho/da/imagem.jpg \
  --class-names artifacts/reports/mobilenet_full_metadata.json
```

## Docker com GPU

O treino final foi executado em Docker com GPU NVIDIA. Antes de usar Docker GPU, o host deve
passar nestes testes:

```bash
nvidia-smi
nvidia-ctk cdi list
docker run --rm --gpus all nvidia/cuda:12.5.1-base-ubuntu22.04 nvidia-smi
```

Comando usado para treinar com limites conservadores de recursos:

```bash
docker run --rm --name plant-village-train-gpu \
  --gpus all \
  --cpus=2 \
  --memory=8g \
  --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/app" \
  -w /app \
  -e HOME=/tmp \
  -e PYTHONPATH=/app/src \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e KERAS_HOME=/app/.keras \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  tensorflow/tensorflow:2.16.1-gpu \
  bash -lc 'python -m pip install --user -q "protobuf<5" tensorflow-datasets pandas scikit-learn seaborn importlib-resources && python -m plant_village.train --model mobilenet --run-name mobilenet_full --epochs 20 --batch-size 8'
```

Comando usado para avaliar:

```bash
docker run --rm --name plant-village-eval-gpu \
  --gpus all \
  --cpus=2 \
  --memory=8g \
  --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/app" \
  -w /app \
  -e HOME=/tmp \
  -e PYTHONPATH=/app/src \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e KERAS_HOME=/app/.keras \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  tensorflow/tensorflow:2.16.1-gpu \
  bash -lc 'python -m pip install --user -q "protobuf<5" tensorflow-datasets pandas scikit-learn seaborn importlib-resources && python -m plant_village.evaluate artifacts/models/mobilenet_full.keras'
```

## Metodologia

1. Carregar o PlantVillage completo via TensorFlow Datasets.
2. Usar split deterministico com seed fixa:
   - 70% treino
   - 15% validacao
   - 15% teste
3. Redimensionar imagens para 224x224.
4. Aplicar aumento de dados apenas no treino:
   - flip horizontal
   - rotacao pequena
   - zoom pequeno
5. Treinar baseline CNN e MobileNetV2.
6. Usar callbacks:
   - checkpoint pelo melhor `val_accuracy`
   - early stopping por `val_loss`
   - reducao de learning rate em plateau
7. Avaliar em conjunto de teste separado com:
   - accuracy
   - precision
   - recall
   - F1
   - matriz de confusao
   - exemplos de acertos e erros

## Boas Praticas Aplicadas

- Codigo organizado como pacote em `src/`.
- Scripts expostos em `pyproject.toml`.
- Configuracao centralizada em `ExperimentConfig`.
- Validacoes de regras do experimento.
- Split deterministico e preservacao da quantidade total de exemplos.
- Pipeline `tf.data` com limites conservadores de CPU, shuffle e prefetch.
- `memory_growth` ativado para GPU.
- Testes automatizados cobrindo regras de negocio e comportamento operacional.
- Artefatos de avaliacao versionados para auditoria do resultado.

## Limitacoes

O PlantVillage possui muitas imagens com folha centralizada, boa iluminacao e fundo controlado.
Por isso, um bom resultado nessa base nao garante o mesmo desempenho em fotos reais de campo.

Melhorias futuras recomendadas:

- Validar com fotos reais fora do PlantVillage.
- Fazer fine tuning controlado de mais camadas da MobileNetV2.
- Testar EfficientNet ou modelos mais recentes.
- Usar data augmentation mais proximo de cenarios reais.
- Medir desempenho por especie e por tipo de doenca.

## Estado do TODO

O arquivo `TODO_PROJETO.md` resume o progresso do projeto. O que falta para uma entrega
academica completa e principalmente transformar os resultados em relatorio final e slides.

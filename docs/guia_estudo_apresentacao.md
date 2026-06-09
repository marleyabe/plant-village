# Guia de estudo da apresentação

Este guia acompanha o arquivo `apresentacao_final.html`. A divisão segue as bolinhas no canto dos slides:

- <span style="color:#2f6f4e">●</span> **Verde - Marley:** slides 1 a 5
- <span style="color:#c99232">●</span> **Dourado - Luis Pedro:** slides 6 a 10
- <span style="color:#b84a3a">●</span> **Vermelho - Maycon:** slides 11 a 14

## Visão geral do projeto

O projeto treina e avalia um classificador de imagens para o dataset PlantVillage. A entrada é uma imagem de folha. A saída é uma das 38 classes, indicando planta saudável ou uma doença específica.

Escolhemos o PlantVillage porque é uma base conhecida, pública e já organizada no TensorFlow Datasets. Isso reduziu o trabalho de limpeza dos dados e permitiu focar no que era mais importante para a disciplina: pipeline de dados, CNN, transfer learning, treino, avaliação, testes e reprodutibilidade.

Resultado principal:

- Dataset: 54.303 imagens
- Classes: 38
- Split: 70% treino, 15% validação, 15% teste
- Treino: 38.012 imagens
- Validação: 8.145 imagens
- Teste: 8.146 imagens
- Modelo final: MobileNetV2 com transfer learning
- Entrada: 224x224x3
- Épocas: 20
- Batch size: 8
- Acurácia no teste: 95,40%
- F1 macro: 94,41%
- F1 ponderado: 95,36%

## <span style="color:#2f6f4e">●</span> Marley - slides 1 a 5

### Ideia principal

Explicar o contexto do projeto, por que a base foi escolhida, qual era o objetivo técnico e como os dados foram preparados.

### Slide 1 - Capa

Pontos para falar:

- O trabalho é uma classificação multiclasse usando deep learning.
- A base é o PlantVillage, com 54.303 imagens e 38 classes.
- O modelo final usa TensorFlow e MobileNetV2.
- O resultado principal foi 95,40% de acurácia no teste.

Frase possível:

> "Nosso projeto usa o PlantVillage para treinar um classificador de folhas em 38 classes. A ideia foi usar uma base conhecida para aplicar um fluxo completo de deep learning, desde o carregamento dos dados até a avaliação final."

### Slide 2 - Escolha do dataset

Pontos para falar:

- O PlantVillage já é conhecido na área.
- Está disponível no TensorFlow Datasets.
- As classes já vêm organizadas.
- Isso deixou o foco no modelo e na avaliação, não em coleta ou limpeza manual.

Evite falar que a base foi escolhida porque "resolve um grande problema do mundo". A motivação aqui é mais técnica e acadêmica.

### Slide 3 - Objetivo técnico

Pontos para falar:

- O objetivo foi criar um classificador reprodutível para 38 classes.
- O projeto não é só um notebook: tem pacote Python, CLI, testes, métricas, modelo e artefatos.
- O fluxo cobre treino, avaliação e predição.

Código relacionado:

```python
bundle = load_datasets(config)
model = build_model("mobilenet", bundle.num_classes, config, fine_tune=False)
```

### Slide 4 - Dataset

Pontos para falar:

- Total: 54.303 imagens.
- Split determinístico: 70/15/15.
- Teste separado: 8.146 imagens que não foram usadas para treinar os pesos.
- O split usa seed fixa, então o experimento pode ser reproduzido.

Se perguntarem por que não usamos cross-validation:

> "Como o dataset é grande e o treino é caro, usamos treino, validação e teste separados. Isso é mais simples, reprodutível e adequado para o escopo do trabalho."

### Slide 5 - Pipeline de dados

Pontos para falar:

- `tfds.load` carrega o PlantVillage.
- `shuffle_files=False` e seed fixa ajudam na reprodutibilidade.
- As imagens são redimensionadas para 224x224.
- O batch e o prefetch foram controlados para não estourar CPU e RAM.
- Data augmentation roda apenas no treino.

Código do slide:

```python
bundle = load_datasets(config)
train_ds = bundle.train
val_ds = bundle.val
```

## <span style="color:#c99232">●</span> Luis Pedro - slides 6 a 10

### Ideia principal

Explicar o modelo, como foi treinado e como interpretamos as métricas e os erros.

### Slide 6 - Arquitetura

Pontos para falar:

- A entrada do modelo é 224x224x3.
- Usamos augmentation com flip horizontal, rotação pequena e zoom pequeno.
- MobileNetV2 foi usada como backbone pré-treinado no ImageNet.
- Removemos o topo original e usamos `GlobalAveragePooling2D`, `Dropout` e `Dense softmax`.
- O backbone ficou congelado no treino principal.

Código do slide:

```python
model = build_model(
    "mobilenet",
    num_classes=38,
    config=config,
    fine_tune=False,
)
```

Resposta curta para "por que MobileNetV2?":

> "Porque é uma CNN pré-treinada, leve e adequada para transfer learning. Ela já aprendeu características visuais gerais no ImageNet, e nós adaptamos o classificador final para as 38 classes do PlantVillage."

### Slide 7 - Treinamento

Pontos para falar:

- Foram 20 épocas.
- Batch size 8.
- Adam com learning rate inicial `1e-3`.
- Loss: `sparse_categorical_crossentropy`.
- Checkpoint pelo melhor `val_accuracy`.
- EarlyStopping monitorando `val_loss`.
- ReduceLROnPlateau reduz o learning rate quando a validação estabiliza.
- Treinamos com Docker e GPU NVIDIA, limitando 2 CPUs e 8 GB de RAM.

Ponto importante:

> "A GPU reduziu bastante o tempo de treino, mas mantivemos limites de CPU e RAM para não sobrecarregar a máquina."

### Slide 8 - Curva de treino

Pontos para falar:

- A validação subiu rápido nas primeiras épocas.
- Depois os ganhos ficaram menores.
- A redução de learning rate ajuda quando o `val_loss` para de melhorar.
- Melhor `val_accuracy`: 0,9596.
- Menor `val_loss`: 0,1319.

Se perguntarem se houve overfitting:

> "Não parece ter sido um overfitting forte pelo resultado no teste. Mas existe diferença entre base controlada e fotos reais, então a generalização externa ainda precisa ser validada."

### Slide 9 - Métricas

Pontos para falar:

- Acurácia mede o acerto global.
- F1 macro dá o mesmo peso para cada classe.
- F1 ponderado considera a quantidade de imagens por classe.
- Acurácia no teste: 95,40%.
- F1 macro: 94,41%.
- F1 ponderado: 95,36%.

Resposta curta para "por que usar F1 se já temos acurácia?":

> "A acurácia resume o desempenho geral, mas pode esconder classes piores. O F1 macro ajuda a ver se o modelo está bem de forma mais equilibrada entre as classes."

### Slide 10 - Matriz de confusão

Pontos para falar:

- A matriz mostra onde o modelo acerta e onde confunde.
- Algumas classes são visualmente parecidas.
- O relatório por classe mostra precision, recall, F1 e suporte.
- Isso ajuda a decidir onde fazer fine tuning ou coletar mais dados.

Resposta curta para "o que fazer para reduzir os erros?":

> "Analisar as classes mais confundidas, reforçar dados dessas classes, testar augmentation mais realista e fazer fine tuning controlado de mais camadas da MobileNetV2."

## <span style="color:#b84a3a">●</span> Maycon - slides 11 a 14

### Ideia principal

Mostrar como o modelo é usado, explicar exemplos de acerto e erro e fechar com limitações e próximos passos.

### Slide 11 - Demonstração

Pontos para falar:

- O script `plant-predict` recebe o modelo `.keras` e uma imagem.
- O metadata mantém a ordem correta das classes.
- A imagem é lida com 3 canais e redimensionada para 224x224.
- `model.predict` retorna 38 probabilidades.
- `argmax` escolhe a classe prevista.
- `max` pega a confiança.

Código do slide:

```python
pred = model.predict(image)
class_id = pred.argmax()
confidence = pred.max()
```

Comando:

```bash
uv run plant-predict artifacts/models/mobilenet_full.keras imagem.jpg \
  --class-names artifacts/reports/mobilenet_full_metadata.json
```

### Slide 12 - Exemplos de acerto

Pontos para falar:

- Os exemplos vêm do conjunto de teste.
- A classe prevista bateu com a classe real.
- Alguns acertos aparecem com confiança muito alta.
- Isso ajuda na demonstração, mas não substitui as métricas globais.

Resposta curta para "por que mostrar exemplos se já temos métricas?":

> "As métricas resumem o comportamento geral. Os exemplos ajudam a visualizar casos concretos e a conferir se a saída do modelo faz sentido."

### Slide 13 - Exemplos de erro

Pontos para falar:

- Mesmo com boa acurácia, o modelo ainda erra.
- Alguns erros aparecem com confiança alta.
- Isso mostra que confiança alta não significa garantia absoluta.
- Erros desse tipo indicam necessidade de analisar classes parecidas e testar imagens fora da base.

Resposta curta para "por que o modelo erra com confiança alta?":

> "Porque a softmax força uma distribuição de probabilidade entre as classes conhecidas. Se a imagem parece com outra classe para o modelo, ele pode errar com alta confiança."

### Slide 14 - Conclusões

Pontos para falar:

- O modelo final chegou a 95,40% de acurácia no teste.
- O projeto ficou organizado e reprodutível.
- O modelo, métricas e gráficos estão versionados.
- A principal limitação é que o PlantVillage é uma base controlada.
- O próximo passo seria validar com fotos reais fora do PlantVillage.

Fechamento possível:

> "O principal resultado não é só a acurácia. O projeto entrega um pipeline completo e reprodutível, com modelo treinado, avaliação, artefatos e análise dos limites."

## Perguntas prováveis do professor

### 1. Por que vocês escolheram o PlantVillage?

Porque é uma base conhecida, pública e disponível no TensorFlow Datasets. As imagens já estão organizadas em classes, o que permitiu focar em deep learning, transfer learning, avaliação e reprodutibilidade.

### 2. Quantas classes existem no problema?

São 38 classes. Elas combinam espécie da planta e condição da folha, por exemplo `Tomato___Late_blight` ou `Apple___healthy`.

### 3. Quantas imagens foram usadas?

Foram 54.303 imagens no total. O split foi 70/15/15:

- 38.012 treino
- 8.145 validação
- 8.146 teste

### 4. O conjunto de teste foi usado no treino?

Não. O teste foi separado antes da avaliação final. Ele foi usado apenas para medir o desempenho depois do treino.

### 5. Por que usar MobileNetV2 em vez de treinar uma CNN do zero?

MobileNetV2 já vem pré-treinada no ImageNet e aprende características visuais gerais. Com transfer learning, o treino fica mais eficiente e tende a performar melhor do que uma CNN simples treinada do zero, principalmente em um projeto acadêmico com tempo limitado.

### 6. O que significa transfer learning?

É reaproveitar uma rede já treinada em uma base grande e adaptar o classificador final para um novo problema. Neste projeto, usamos a MobileNetV2 como extratora de características e treinamos a saída para 38 classes.

### 7. O backbone foi treinado?

No treino principal, não. O backbone ficou congelado (`fine_tune=False`). O projeto suporta fine tuning por flag, mas o resultado final apresentado usa o backbone congelado.

### 8. Por que a entrada é 224x224?

Porque é um tamanho padrão para modelos pré-treinados como MobileNetV2. Redimensionar todas as imagens para 224x224 também padroniza o batch de entrada.

### 9. O que é data augmentation?

É aplicar transformações nas imagens de treino, como flip, rotação e zoom, para aumentar a variação visual e reduzir overfitting. No projeto, augmentation é aplicado apenas no treino.

### 10. Por que augmentation não roda na validação e no teste?

Porque validação e teste devem medir o desempenho em dados fixos. Se aplicássemos augmentation nesses conjuntos, a avaliação poderia variar e ficaria menos reprodutível.

### 11. Qual loss foi usada?

`sparse_categorical_crossentropy`, porque o problema é multiclasse e os rótulos são inteiros, não one-hot encoded.

### 12. Qual otimizador foi usado?

Adam com learning rate inicial `1e-3`.

### 13. Quantas épocas foram usadas?

20 épocas no experimento final.

### 14. Por que batch size 8?

Foi uma escolha conservadora para caber na GPU disponível e reduzir risco de uso excessivo de memória. Como o treino foi feito em Docker com limites de recurso, batch 8 foi mais seguro.

### 15. O que significa acurácia de 95,40%?

Significa que, no conjunto de teste com 8.146 imagens, o modelo acertou aproximadamente 95,40% das previsões.

### 16. O que é F1 macro?

É a média do F1 calculado por classe, dando o mesmo peso para todas as classes. Ele ajuda a avaliar se o modelo performa bem de forma equilibrada.

### 17. O que é F1 ponderado?

É o F1 calculado considerando o suporte de cada classe. Classes com mais imagens têm peso maior no resultado final.

### 18. Por que F1 macro é menor que F1 ponderado?

Isso pode indicar que algumas classes menores ou mais difíceis tiveram desempenho um pouco pior. O F1 ponderado fica mais próximo da acurácia porque considera a distribuição de exemplos.

### 19. O modelo generaliza para fotos reais de campo?

Ainda não dá para garantir. O PlantVillage é uma base controlada, com folha centralizada, boa iluminação e fundo mais padronizado. Para afirmar generalização externa, seria necessário testar com fotos reais fora da base.

### 20. Qual é a principal limitação do projeto?

A principal limitação é a diferença entre o PlantVillage e cenários reais. O resultado é forte dentro da base, mas ainda precisa ser validado com imagens capturadas em condições menos controladas.

### 21. O que a matriz de confusão mostra?

Mostra quais classes o modelo acerta e quais classes ele confunde. Ela ajuda a identificar padrões de erro e classes que precisam de mais análise.

### 22. Por que alguns erros têm confiança alta?

Porque a saída softmax sempre distribui probabilidade entre as classes conhecidas. Se o modelo interpreta a imagem como muito parecida com uma classe errada, ele pode atribuir alta confiança a essa classe.

### 23. O que vocês fariam para melhorar o modelo?

Próximos passos:

- Validar com fotos reais fora do PlantVillage.
- Fazer fine tuning controlado de mais camadas.
- Testar EfficientNet ou outro backbone.
- Usar augmentation mais próximo de cenários reais.
- Analisar as classes com maior confusão.

### 24. Como vocês garantiram reprodutibilidade?

Usamos seed fixa, split determinístico, `shuffle_files=False`, configuração centralizada em `ExperimentConfig`, metadados salvos do treino, modelo versionado e artefatos de avaliação no repositório.

### 25. O que existe no GitHub além do HTML da apresentação?

O repositório contém código do pacote, scripts CLI, testes, modelo `.keras`, métricas, gráficos, matriz de confusão, exemplos de acertos e erros, documentação e arquivos de configuração.

### 26. Que testes foram feitos?

Os testes automatizados cobrem regras do experimento, validação de configuração, split de dados, comportamento operacional e pontos críticos para evitar configurar o treino de forma inválida.

### 27. Por que colocar o modelo no Git?

O arquivo `.keras` tem cerca de 10 MB, então é pequeno o suficiente para versionar. Isso facilita a entrega acadêmica, permite reproduzir a avaliação sem retreinar e deixa o repositório completo.

### 28. Por que usar Docker?

Docker ajudou a isolar o ambiente, usar a imagem TensorFlow GPU e limitar recursos como CPU e RAM. Isso tornou o treino mais controlado.

### 29. Por que usar GPU?

Treino de CNN é caro em CPU. A GPU acelera operações matriciais e convolucionais. No experimento, a primeira época caiu de cerca de 944s na CPU para cerca de 111s na GPU.

### 30. Qual é a mensagem final do projeto?

O projeto mostra um fluxo completo e reprodutível de deep learning em uma base conhecida: dados, modelo, treino, avaliação, predição, métricas, testes e artefatos versionados.

## Checklist de treino antes da apresentação

- Cada pessoa deve saber explicar seus slides sem ler texto inteiro.
- Cada pessoa deve saber os números principais: 54.303 imagens, 38 classes, 20 épocas, batch 8, 95,40% de acurácia.
- Todos devem saber explicar a limitação da base controlada.
- Todos devem saber responder por que MobileNetV2 foi escolhida.
- Todos devem saber diferenciar acurácia, F1 macro e F1 ponderado.
- Todos devem saber dizer que o teste não foi usado no treino.
- Ensaiar uma versão de 10 minutos e outra de 15 minutos.


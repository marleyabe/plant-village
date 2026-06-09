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

## Pegadinhas e perguntas além dos slides

Esta seção remove perguntas óbvias que já estão respondidas diretamente nos slides, como quantidade de classes, épocas ou acurácia. O foco aqui é treinar respostas para perguntas que exigem justificativa técnica.

### 1. A acurácia alta prova que o modelo funciona em fotos reais?

Não. A acurácia alta prova bom desempenho no teste do PlantVillage. Como a base é controlada, com folha centralizada, boa iluminação e fundo mais padronizado, ainda precisamos validar com fotos reais fora da base para falar em generalização para campo.

### 2. Existe risco de vazamento de dados no split?

O split foi feito de forma determinística e o teste ficou separado do treino. O principal ponto de atenção é que o PlantVillage pode ter imagens muito parecidas dentro da mesma base. Então, mesmo sem usar o teste no treino, a validação externa ainda é importante.

### 3. Por que não usar cross-validation?

Cross-validation aumentaria muito o custo porque exigiria treinar o modelo várias vezes. Para uma base com 54.303 imagens e treino com CNN, o split fixo 70/15/15 é uma escolha mais simples, reprodutível e adequada para o escopo do trabalho.

### 4. Por que o F1 macro ficou menor que o F1 ponderado?

O F1 macro dá o mesmo peso para todas as classes. Se algumas classes menores ou mais difíceis tiverem desempenho pior, ele cai mais. O F1 ponderado considera o número de exemplos por classe, por isso tende a ficar mais próximo da acurácia.

### 5. Se o modelo erra com 99% de confiança, a confiança não serve?

Serve, mas precisa ser interpretada com cuidado. A softmax distribui probabilidade apenas entre as classes conhecidas. Se a imagem parece muito com uma classe errada para o modelo, ele pode errar com confiança alta. Isso não é uma garantia de certeza real.

### 6. Por que congelar o backbone em vez de fazer fine tuning direto?

Congelar o backbone reduz custo, risco de overfitting e instabilidade no treino. Como a MobileNetV2 já vem pré-treinada, primeiro treinamos o classificador final. Fine tuning pode ser um próximo passo, mas deve ser feito com learning rate menor e controle por validação.

### 7. O que poderia dar errado ao fazer fine tuning?

Se destravarmos muitas camadas com learning rate alto, o modelo pode perder parte dos pesos úteis aprendidos no ImageNet ou overfitar ao PlantVillage. O ideal seria destravar poucas camadas finais, reduzir o learning rate e comparar no conjunto de validação.

### 8. Por que augmentation simples pode não ser suficiente?

Flip, rotação e zoom ajudam, mas fotos reais podem ter variações maiores: sombra, fundo complexo, folha parcialmente cortada, blur, ângulo diferente e iluminação ruim. Para aproximar uso real, seria necessário augmentation mais forte e validação fora do PlantVillage.

### 9. O modelo classifica a doença ou só reconhece padrões da base?

Tecnicamente, ele aprende padrões visuais associados às classes do dataset. Se a base tiver viés de fundo, iluminação ou enquadramento, o modelo pode usar esses sinais também. Por isso a validação externa é essencial.

### 10. Por que não basta mostrar exemplos de acerto?

Exemplos de acerto ajudam na demonstração, mas podem ser selecionados e não representam o comportamento geral. O resultado precisa ser sustentado por métricas no teste, matriz de confusão e análise de erros.

### 11. Como vocês sabem que o modelo não decorou o treino?

Não dá para afirmar só olhando o treino. A evidência contra memorização forte é o desempenho em validação e teste separados. Mesmo assim, como o teste vem da mesma base, a prova mais forte seria avaliar com imagens externas.

### 12. Por que usar `sparse_categorical_crossentropy` e não `categorical_crossentropy`?

Porque os rótulos do dataset são inteiros. `categorical_crossentropy` seria mais adequada se os rótulos estivessem em one-hot encoding. Usar `sparse_categorical_crossentropy` evita uma transformação desnecessária.

### 13. O que aconteceria se data augmentation fosse aplicado no teste?

O teste deixaria de ser um conjunto fixo e a métrica poderia variar conforme as transformações. Isso prejudica a comparação e a reprodutibilidade. Por isso augmentation fica apenas no treino.

### 14. Por que salvar metadata junto com o modelo?

O metadata guarda informações como classes, tamanho da imagem, batch e seed. Isso evita avaliar ou predizer com configuração diferente da usada no treino, principalmente a ordem das classes.

### 15. O que garante que a ordem das classes está correta na predição?

O script de predição usa o metadata salvo. Sem isso, poderíamos carregar o modelo corretamente, mas interpretar o índice de saída com uma lista de classes em ordem errada.

### 16. Por que o batch size foi tão baixo?

Foi uma escolha conservadora para caber na GPU e evitar uso excessivo de memória. O foco era concluir o experimento completo com estabilidade, não maximizar throughput.

### 17. A GPU muda o resultado do modelo?

Em princípio, a GPU muda principalmente a velocidade. Pequenas diferenças numéricas podem acontecer por operações paralelas, mas a configuração com seed e split fixos ajuda a manter o experimento controlado.

### 18. O que vocês fariam se a matriz de confusão mostrasse erro concentrado em poucas classes?

Olharíamos essas classes no relatório por classe, revisaríamos exemplos de erro, testaríamos augmentation mais específico, consideraríamos fine tuning e, se possível, adicionaríamos dados externos dessas classes.

### 19. O modelo sabe lidar com uma planta ou doença que não está nas 38 classes?

Não de forma confiável. O modelo sempre escolhe uma das 38 classes conhecidas. Para lidar com classes desconhecidas, seria necessário pensar em limiar de confiança, detecção de out-of-distribution ou treinar com dados desse novo caso.

### 20. Por que versionar o modelo no Git não é sempre uma boa prática?

Modelos grandes podem deixar o repositório pesado. Neste projeto, o `.keras` tem cerca de 10 MB, então é aceitável para entrega acadêmica. Em projetos maiores, seria melhor usar Git LFS, storage externo ou registry de modelos.

### 21. O que os testes automatizados realmente protegem?

Eles protegem regras do experimento, como configuração inválida, split inconsistente e comportamento operacional. Eles não provam que o modelo é bom; eles reduzem risco de erro no pipeline.

### 22. Se fossem refazer o projeto com mais tempo, qual seria a melhoria mais importante?

Validar com imagens reais fora do PlantVillage. Depois disso, faria sentido comparar backbones, fazer fine tuning progressivo e analisar melhor as classes com maior confusão.

## Checklist de treino antes da apresentação

- Cada pessoa deve saber explicar seus slides sem ler texto inteiro.
- Cada pessoa deve saber os números principais: 54.303 imagens, 38 classes, 20 épocas, batch 8, 95,40% de acurácia.
- Todos devem saber explicar a limitação da base controlada.
- Todos devem saber responder por que MobileNetV2 foi escolhida.
- Todos devem saber diferenciar acurácia, F1 macro e F1 ponderado.
- Todos devem saber dizer que o teste não foi usado no treino.
- Ensaiar uma versão de 10 minutos e outra de 15 minutos.

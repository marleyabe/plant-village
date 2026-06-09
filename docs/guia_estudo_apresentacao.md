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

## Pegadinhas de Deep Learning

Esta seção foca em perguntas que podem aparecer por causa da matéria de deep learning. A ideia é treinar respostas que exigem entender o modelo, e não apenas repetir números dos slides.

### 1. Por que uma CNN faz sentido para imagens de folhas?

CNNs são adequadas para imagem porque exploram padrões locais, como bordas, texturas, manchas e formatos. As convoluções aprendem filtros que detectam características visuais em diferentes regiões da imagem. Em folhas, isso é útil porque doenças aparecem como padrões visuais de cor, textura e lesão.

### 2. O que a MobileNetV2 está aprendendo nesse projeto?

Como o backbone foi pré-treinado no ImageNet, ele já traz filtros úteis para detectar formas, bordas, texturas e padrões visuais gerais. No nosso projeto, o classificador final aprende a combinar essas características para separar as 38 classes do PlantVillage.

### 3. Por que usar transfer learning e não treinar tudo do zero?

Treinar do zero exigiria mais dados, mais tempo e mais ajuste fino. Com transfer learning, partimos de uma rede que já aprendeu representações visuais gerais. Isso acelera o treino e costuma melhorar o resultado, principalmente em projetos com prazo e recurso limitados.

### 4. Qual é a diferença entre feature extractor e fine tuning?

Como feature extractor, a MobileNetV2 fica congelada e só o topo classificador é treinado. No fine tuning, algumas camadas do backbone também são destravadas para adaptar melhor os filtros ao novo dataset. Fine tuning pode melhorar o resultado, mas aumenta risco de overfitting.

### 5. Por que congelar o backbone no treino principal?

Congelar o backbone reduz custo computacional, deixa o treino mais estável e diminui risco de destruir pesos úteis aprendidos no ImageNet. É uma primeira etapa comum em transfer learning: treina-se o classificador final antes de pensar em destravar camadas.

### 6. O que poderia dar errado no fine tuning?

Se destravarmos muitas camadas ou usarmos learning rate alto, o modelo pode overfitar ao PlantVillage ou perder representações gerais úteis do ImageNet. O ajuste correto seria destravar poucas camadas finais, usar learning rate menor e acompanhar validação.

### 7. Por que a saída usa softmax?

Softmax transforma os logits em uma distribuição de probabilidade entre as 38 classes. Como o problema é multiclasse e cada imagem pertence a uma classe, softmax é adequado para escolher a classe mais provável.

### 8. Softmax alto significa que o modelo tem certeza?

Não necessariamente. Softmax alto significa que, entre as classes conhecidas, uma classe recebeu probabilidade muito maior. O modelo pode errar com 99% de confiança se aprendeu um padrão errado ou se a imagem parece muito com outra classe.

### 9. Por que usar `sparse_categorical_crossentropy`?

Porque temos um problema multiclasse com rótulos inteiros. Se os rótulos fossem one-hot encoded, usaríamos `categorical_crossentropy`. A versão `sparse` evita converter os labels para one-hot sem mudar o objetivo matemático da classificação.

### 10. O que a loss mede durante o treino?

A loss mede o quanto a distribuição prevista pelo modelo está distante da classe correta. Mesmo quando a acurácia parece boa, a loss pode indicar se o modelo está ficando confiante demais em previsões erradas ou se ainda há espaço de melhoria.

### 11. Como identificar overfitting pela curva de treino?

Um sinal clássico é a acurácia de treino continuar subindo enquanto a validação para de melhorar ou piora. Outro sinal é a loss de treino cair enquanto a loss de validação sobe. No nosso caso, o resultado no teste sugere que não houve overfitting forte dentro do PlantVillage.

### 12. Por que data augmentation ajuda contra overfitting?

Augmentation cria variações das imagens de treino, como flip, rotação e zoom. Isso força o modelo a aprender padrões mais robustos, em vez de decorar exatamente o enquadramento das imagens originais.

### 13. Por que augmentation não deve ser usado no teste?

O teste precisa medir o modelo em um conjunto fixo e comparável. Se aplicarmos transformações aleatórias no teste, a métrica pode mudar de uma execução para outra e deixa de representar uma avaliação estável.

### 14. O modelo pode estar aprendendo o fundo em vez da doença?

Pode. Redes neurais aprendem correlações presentes nos dados, não necessariamente a causa real. Se o fundo, iluminação ou enquadramento estiverem associados a uma classe, o modelo pode usar esses sinais. Por isso a validação fora do PlantVillage é importante.

### 15. A acurácia alta prova generalização?

Prova generalização apenas para o teste separado da mesma base. Não prova generalização para fotos reais de campo, porque a distribuição das imagens pode mudar. Esse problema é chamado de shift de distribuição.

### 16. O que é shift de distribuição nesse contexto?

É quando as imagens de uso real são diferentes das imagens de treino e teste. Por exemplo: folhas no campo, fundo complexo, iluminação ruim, sombra, blur, ângulo diferente ou folha parcialmente cortada.

### 17. Por que F1 macro é relevante em deep learning multiclasse?

Porque acurácia pode esconder desempenho ruim em classes menos frequentes. O F1 macro dá o mesmo peso para cada classe, então ajuda a perceber se o modelo está aprendendo todas as classes ou só indo bem nas maiores.

### 18. O que significa erro concentrado em algumas classes?

Pode indicar que as classes têm sintomas visualmente parecidos, poucos exemplos, ruído nos rótulos ou que o modelo não aprendeu características suficientes para separá-las. A matriz de confusão ajuda a identificar esses grupos.

### 19. O que é dropout e por que ele aparece no topo do modelo?

Dropout desativa aleatoriamente parte das ativações durante o treino. Isso reduz dependência excessiva de neurônios específicos e ajuda a regularizar o classificador final.

### 20. Por que usar GlobalAveragePooling2D antes da Dense?

Ele resume cada mapa de características em um valor médio, reduzindo bastante a quantidade de parâmetros em comparação com `Flatten`. Isso deixa o topo mais leve e reduz risco de overfitting.

### 21. O que acontece se o learning rate for alto demais?

O treino pode ficar instável, a loss pode oscilar ou o modelo pode não convergir. Em fine tuning, learning rate alto é ainda mais perigoso porque pode alterar demais os pesos pré-treinados.

### 22. O que acontece se o learning rate for baixo demais?

O treino pode ficar muito lento ou parar em uma solução pior. Por isso usamos `ReduceLROnPlateau`: quando a validação estabiliza, o learning rate é reduzido para permitir ajustes menores.

### 23. Por que batch size influencia o treino?

Batch size define quantas imagens entram em cada atualização de gradiente. Batches menores usam menos memória e podem introduzir mais ruído no gradiente. Batches maiores tendem a ser mais estáveis, mas exigem mais memória.

### 24. O modelo sabe lidar com uma classe que não existe no treino?

Não de forma confiável. Como a saída softmax sempre escolhe uma das 38 classes, uma imagem fora dessas classes ainda será forçada para alguma classe conhecida. Para lidar com desconhecidos, precisaríamos de limiar de confiança ou técnicas de out-of-distribution.

### 25. Se fossem melhorar o projeto com foco em deep learning, o que fariam?

Primeiro validaríamos com imagens reais fora do PlantVillage. Depois testaríamos fine tuning progressivo, augmentation mais realista, comparação com outro backbone como EfficientNet e análise das classes com maior confusão.

## Checklist de treino antes da apresentação

- Cada pessoa deve saber explicar seus slides sem ler texto inteiro.
- Cada pessoa deve saber os números principais: 54.303 imagens, 38 classes, 20 épocas, batch 8, 95,40% de acurácia.
- Todos devem saber explicar a limitação da base controlada.
- Todos devem saber responder por que MobileNetV2 foi escolhida.
- Todos devem saber diferenciar acurácia, F1 macro e F1 ponderado.
- Todos devem saber dizer que o teste não foi usado no treino.
- Ensaiar uma versão de 10 minutos e outra de 15 minutos.

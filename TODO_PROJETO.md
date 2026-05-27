# Classificação de Doenças em Folhas de Plantas com Deep Learning

## Equipe

- Marley Abe Silva
- Luis Pedro de Castro Alves
- Maycon Moriy Abe Machado

## Dataset

**Base escolhida:** PlantVillage  
**Link:** https://tensorflow.google.cn/datasets/catalog/plant_village

O PlantVillage possui imagens de folhas saudáveis e doentes, organizadas por espécie da planta e tipo de doença. A versão disponível no TensorFlow Datasets contém 54.303 imagens distribuídas em 38 classes.

## Descrição do projeto

O trabalho consiste em treinar uma rede neural convolucional com TensorFlow para classificar imagens de folhas de plantas. A entrada do sistema será uma imagem de folha; a saída será a classe prevista, indicando se a folha está saudável ou qual doença foi identificada.

Para manter o escopo viável, o projeto pode usar apenas um subconjunto da base, por exemplo classes de tomate ou batata. Essa escolha reduz o tempo de treinamento e facilita a análise dos erros.

## Objetivo

Desenvolver e avaliar um classificador de imagens capaz de reconhecer doenças em folhas de plantas usando técnicas de Deep Learning.

## Todo List

### 1. Ambiente

- [ ] Criar o ambiente Python do projeto.

### 2. Dataset

- [ ] Baixar e organizar o PlantVillage em treino, validação e teste.

### 3. Pré-processamento

- [ ] Padronizar as imagens e aplicar aumento de dados quando necessário.

### 4. Modelo

- [ ] Definir e implementar a rede convolucional em TensorFlow.

### 5. Treinamento

- [ ] Treinar o classificador e registrar o histórico de perda e acurácia.

### 6. Avaliação

- [ ] Avaliar o modelo com métricas, matriz de confusão e exemplos de acertos e erros.

### 7. Análise

- [ ] Interpretar os resultados, apontar limitações da base e sugerir melhorias.

### 8. Entrega

- [ ] Organizar código, relatório, gráficos e slides da apresentação.

## Resultados esperados

- Modelo treinado para classificar folhas saudáveis e doentes.
- Métricas de desempenho no conjunto de teste.
- Matriz de confusão.
- Análise dos principais erros.
- Relatório com metodologia, resultados e limitações.

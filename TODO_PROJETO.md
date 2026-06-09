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

O experimento principal deve usar o máximo possível da base PlantVillage, preferencialmente as 54.303 imagens e 38 classes. Um subconjunto reduzido pode ser usado apenas durante o desenvolvimento para validar rapidamente o código, mas os resultados finais devem priorizar a base completa.

## Objetivo

Desenvolver e avaliar um classificador de imagens capaz de reconhecer doenças em folhas de plantas usando técnicas de Deep Learning.

## Todo List

### 1. Ambiente

- [x] Criar a configuracao do ambiente Python do projeto com uv.

### 2. Dataset

- [x] Baixar e organizar o PlantVillage completo em treino, validação e teste.

### 3. Pré-processamento

- [x] Implementar padronizacao das imagens e aumento de dados no pipeline.

### 4. Modelo

- [x] Definir e implementar CNN baseline e MobileNetV2 com transfer learning.

### 5. Treinamento

- [x] Treinar o classificador e registrar o histórico de perda e acurácia.

### 6. Avaliação

- [x] Implementar avaliação com métricas, matriz de confusão e exemplos de acertos e erros.

### 7. Análise

- [x] Interpretar os resultados, apontar limitações da base e sugerir melhorias.

### 8. Entrega

- [x] Organizar código, modelo treinado, métricas e gráficos no repositório.
- [ ] Preparar relatório textual final e slides da apresentação.

## Resultados esperados

- Modelo treinado para classificar folhas saudáveis e doentes.
- Métricas de desempenho no conjunto de teste.
- Matriz de confusão.
- Análise dos principais erros.
- Relatório com metodologia, resultados e limitações.

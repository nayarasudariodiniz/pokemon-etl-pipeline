# ETL - Pipeline de Engenharia de Dados

![Status](https://img.shields.io/badge/STATUS-CONCLUÍDO-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB4223?style=for-the-badge&logo=xgboost&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)

Este projeto implementa um pipeline completo de **Engenharia de Dados (ETL)** para ingerir dados da **PokéAPI**, transformá-los e alimentar um modelo de Machine Learning (**XGBoost**) capaz de prever o vencedor de batalhas entre Pokémons.

## ⚙️ Arquitetura do Pipeline

O projeto foi estruturado em três etapas principais:

1.  **Extração (Extract):**
    * Consumo da PokéAPI utilizando Python (`requests`).
    * Implementação de lógica de resiliência (retries e backoff exponencial).
    * Salvamento dos dados brutos (JSON) em um Datalake local (`data/raw`) para garantir a imutabilidade.

2.  **Transformação e Carga (Transform & Load):**
    * Tratamento e normalização dos dados brutos.
    * Modelagem e carga em um banco de dados relacional **SQLite**.
    * **Feature Engineering:** Criação de variáveis decisivas para o modelo, como "Vantagem de Tipo" (baseada na tabela de danos) e diferencial de status (Ataque, Defesa, Velocidade).

3.  **Machine Learning:**
    * Treinamento de um classificador **XGBoost**.
    * Integração com dados históricos de batalhas.
    * Pré-processamento com `StandardScaler` e pipeline de inferência.

## 📂 Estrutura dos Scripts

* `extraction.py`: Orquestra a extração da API e carga inicial no Datalake/Banco.
* `prepare_ml_data.py`: Realiza a engenharia de atributos (features) e prepara o dataset de treino.
* `train_model.py`: Treina o modelo, avalia a performance e serializa os artefatos (`.pkl`).

## 🚀 Como Executar

Clone o repositório e instale as dependências:

```bash
  pip install -r requirements.txt
```

Execute o pipeline na ordem:
# 1. Extração de dados da API e carga no SQLite

    python extraction.py 
# 2. Processamento e criação de features para ML
    python prepare_ml_data.py
    
# 3. Treinamento do modelo preditivo
    python train_model.py



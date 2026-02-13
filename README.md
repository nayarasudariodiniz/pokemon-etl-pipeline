# ETL - Pipeline de Engenharia de Dados

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Postman](https://img.shields.io/badge/Postman-FF6C37?style=for-the-badge&logo=postman&logoColor=white)
![Status](https://img.shields.io/badge/STATUS-EM%20DESENVOLVIMENTO-green?style=for-the-badge)

Este projeto é focado na construção de um pipeline de **Engenharia de Dados** completo, realizando o processo de ETL (Extract, Transform, Load) a partir da **PokéAPI**. O projeto é desenvolvido de forma autoral, adaptando conceitos de integração de dados para um cenário de análise de atributos de Pokémon.

## 📋 Objetivo do Projeto
O objetivo é automatizar a ingestão de dados de Pokémon, criando um **Data Lake** local para armazenamento de dados brutos e integrando-os em um banco de dados relacional (SQLite) para alimentar um futuro modelo de Machine Learning focado em predição de batalhas.

## 🛠️ Tecnologias Utilizadas
* **Python**: Automação do pipeline.
* **Postman**: Exploração e documentação da API.
* **SQLite**: Armazenamento dos dados processados.
* **Pandas**: Manipulação e limpeza de dados.

## 📁 Estrutura do Datalake
* **data/raw/**: Armazena os arquivos JSON brutos obtidos da API, garantindo a imutabilidade do dado original.
* **data/database/**: Contém o banco de dados SQLite com os dados processados e prontos para consulta.

## 🚀 Como Executar
1. **Ambiente Virtual**: Crie e ative o ambiente virtual (`venv`) para isolar as dependências.
2. **Dependências**: Instale as bibliotecas necessárias (como `requests`).
3. **Criação do Banco**: Antes da carga, execute o script de criação do banco de dados para gerar as tabelas localmente.
4. **Visualização**: O projeto conta com scripts para visualização de dados em tabelas para validar a integração.

## 🧠 Aprendizados de Percurso
* A exploração via Postman permitiu mapear as estruturas aninhadas do JSON antes da codificação.
* A arquitetura de Datalake local permite reprocessar dados sem gerar novas chamadas desnecessárias à API.
* A adaptação de um projeto de curso para um tema autoral fortalece a compreensão da lógica de integração de dados.

---
*Este README é atualizado conforme o avanço dos módulos do curso e do desenvolvimento do pipeline.*

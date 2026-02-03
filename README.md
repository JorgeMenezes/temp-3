# temp-hackathon
Este repositório é para a equipe 1 da hackathon de 2026.

O desafio da hackathon é descrito abaixo:

# Contexto lúdico
Carla é analista de negócios e trabalha diariamente com planilhas e relatórios, conhece bem os dados da sua área, mas não sabe programar.
Carla tem hipóteses sobre padrões nos dados que poderiam gerar previsões úteis para o negócio. Mas cada novo estudo exige um esforço considerável: explorar os dados, preparar variáveis, testar diferentes abordagens analíticas e comparar resultados. Muitas vezes, esse processo se repete com pequenas variações, consumindo tempo que poderia ser dedicado à interpretação dos resultados e à geração de insights.
Carla deseja uma ferramenta que permita experimentar modelos preditivos por conta própria, sem escrever código. Fazer upload da sua planilha, explorar os dados visualmente, criar transformações,  treinar modelos para validar suas hipóteses e disponibilizá-los para integrações de forma autônoma.


# Desafio
Desenvolver uma plataforma de AutoML que permita a pessoa analista fazer upload de datasets, análise exploratória e processamento de dados, treinar automaticamente múltiplos algoritmos de Machine Learning - classificação, regressão e séries temporais -, comparar seus resultados através de métricas e realizar o deploy do melhor modelo - disponibilizando-o via API REST ou processamento em lote.

# Requisitos mínimos (no final, desejamos ter diferenciais além dos requisitos mínimos)
A plataforma deve aceitar datasets em formatos CSV e Excel	
Permitir visualização e análise exploratória do dataset (estatísticas, distribuições, correlações)
Permitir que o usuário crie e  modifique colunas, trate valores nulos e faça transformações nos dados
Capacidade de treinar múltiplos algoritmos (mínimo de 3 por tipo de problema) sem intervenção manual.
Exibir métricas de performance e ranquear os modelos de forma clara e comparativa
Disponibilizar o modelo treinado via API Rest e processamento em batch	
Interface intuitiva que permita ao usuário completar o fluxo com o mínimo de cliques
Apresentação clara e concisa da solução
Documentação completa explicando arquitetura e como utilizar a plataforma

# Pequenas descrições
Na pasta dados/exemplo-entrada, há um dataset de exemplo para cada um dos tipos de predição que pode ser feita (classificação, regressão e séries temporais). No dataset "online_shoppers.csv", por exemplo, a coluna target é a "Revenue". Na "insurance.scv", a target é a "charges".

Este projeto deve usar Python 3.11

Seria muito interessante que utilizássemos python para o backend e o frontend também (dado que os participantes tem mais familiaridade com python).

# Arquitetura proposta
- **Backend (FastAPI)**: API REST para upload, exploração, transformações, treino AutoML e deploy.
- **Frontend (Streamlit)**: interface web focada no fluxo de analista (upload → exploração → transformações → treino → deploy).
- **Persistência local**: arquivos CSV e artefatos de modelos em `storage/`.

## Fluxo do usuário
1. Upload de dataset CSV/Excel.
2. Exploração automática (amostras, estatísticas, correlações e nulos).
3. Transformações simples (remover colunas, preencher nulos e criar colunas via expressão pandas).
4. Treino AutoML para classificação, regressão ou séries temporais.
5. Ranking dos modelos e deploy via API (predição online ou batch).

# Como executar
## Instalação
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Subir o backend
```bash
uvicorn backend.main:app --reload
```

## Subir o frontend
```bash
streamlit run frontend/app.py
```

# Endpoints principais
- `POST /datasets/upload`: upload CSV/Excel.
- `GET /datasets`: lista datasets.
- `GET /datasets/{id}/summary`: estatísticas, correlações e nulos.
- `POST /datasets/{id}/transform`: transformações (drop, fillna, add_columns).
- `POST /train`: treino AutoML (classificação, regressão, séries temporais).
- `POST /models/{id}/predict`: predição via API.
- `POST /batch/predict?model_id=...`: predição em lote.

# Diferenciais inclusos
- Ranking automático por métricas adequadas (F1 para classificação, RMSE para regressão/séries temporais).
- Relatórios de treino salvos automaticamente em `storage/models`.
- Suporte a séries temporais com ARIMA, SARIMAX e Holt-Winters.

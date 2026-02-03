# backend/app/services/automl_service.py

"""
Serviço de AutoML usando PyCaret
Centraliza toda a lógica de treinamento automático
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import joblib
import json
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class AutoMLService:
    """
    Serviço de Machine Learning Automatizado usando PyCaret.
    Suporta Classificação, Regressão e Séries Temporais.
    """
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.current_setup = None
        self.trained_models = {}
        
    # ==================== CLASSIFICAÇÃO ====================
    
    def train_classification(
        self,
        df: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        top_n_models: int = 5
    ) -> Dict[str, Any]:
        """
        Treina múltiplos modelos de classificação automaticamente.
        
        Args:
            df: DataFrame com os dados
            target: Nome da coluna alvo
            feature_columns: Lista de colunas features (None = todas)
            test_size: Proporção do conjunto de teste
            top_n_models: Número de melhores modelos a retornar
            
        Returns:
            Dicionário com resultados do treinamento
        """
        from pycaret.classification import (
            setup, compare_models, pull, 
            predict_model, save_model, get_config
        )
        
        logger.info(f"Iniciando treinamento de classificação. Target: {target}")
        
        # Filtrar colunas se especificado
        if feature_columns:
            columns_to_use = feature_columns + [target]
            df = df[columns_to_use]
        
        # Gerar ID único para este treinamento
        job_id = str(uuid.uuid4())[:8]
        
        try:
            # ===== SETUP =====
            # O setup faz automaticamente:
            # - Split train/test
            # - Detecta tipos de variáveis
            # - Trata valores nulos
            # - Encoding de categóricas
            # - Normalização (opcional)
            
            logger.info("Executando setup do PyCaret...")
            
            clf_setup = setup(
                data=df,
                target=target,
                train_size=1 - test_size,
                session_id=42,  # Reprodutibilidade
                verbose=False,
                html=False,  # Desabilita output HTML
                
                # Preprocessing automático
                normalize=True,
                normalize_method='zscore',
                remove_multicollinearity=True,
                multicollinearity_threshold=0.6,
                
                # Tratamento de nulos
                imputation_type='simple',
                numeric_imputation='mean',
                categorical_imputation='mode',
                
                # Performance
                n_jobs=-1,  # Usar todos os cores
            )
            
            # ===== COMPARAR MODELOS =====
            logger.info("Comparando modelos...")
            
            # Lista de modelos a comparar (pode customizar)
            models_to_compare = [
                'rf',      # Random Forest
                'xgboost', # XGBoost
                # 'lightgbm',# LightGBM
                # 'lr',      # Logistic Regression
                # 'dt',      # Decision Tree
                # 'knn',     # K-Nearest Neighbors
                # 'nb',      # Naive Bayes
                # 'ada',     # AdaBoost
                # 'gbc',     # Gradient Boosting
                # 'et',      # Extra Trees
            ]
            
            # Treina e compara todos os modelos
            best_model = compare_models(
                include=models_to_compare,
                n_select=top_n_models,
                sort='Accuracy',  # Métrica para ranking
                verbose=False
            )
            
            # ===== OBTER RESULTADOS =====
            results_df = pull()  # DataFrame com métricas de todos os modelos
            
            # ===== SALVAR MELHOR MODELO =====
            model_filename = f"clf_{job_id}"
            model_path = self.models_dir / model_filename
            save_model(best_model, str(model_path))
            
            # Obter informações do setup
            feature_names = get_config('X_train').columns.tolist()

            
            # Preparar resposta
            results = {
                "job_id": job_id,
                "problem_type": "classification",
                "target": target,
                "status": "completed",
                "created_at": datetime.now().isoformat(),
                
                # Informações do dataset
                "dataset_info": {
                    "total_rows": len(df),
                    "train_rows": int(len(df) * (1 - test_size)),
                    "test_rows": int(len(df) * test_size),
                    "n_features": len(feature_names),
                    "features": feature_names
                },
                
                # Resultados dos modelos
                "models_comparison": results_df.to_dict('records'),
                
                # Melhor modelo
                "best_model": {
                    "name": type(best_model).__name__,
                    "model_id": model_filename,
                    "path": str(model_path) + ".pkl",
                    "metrics": results_df.iloc[0].to_dict()
                },
                
                # Ranking
                "ranking": [
                    {
                        "rank": i + "1",
                        "model": row['Model'],
                        "accuracy": row.get('Accuracy', 0),
                        "auc": row.get('AUC', 0),
                        "f1": row.get('F1', 0),
                        "precision": row.get('Prec.', 0),
                        "recall": row.get('Recall', 0)
                    }
                    for i, row in results_df.head(top_n_models).iterrows()
                ]
            }
            
            # Salvar metadata
            metadata_path = self.models_dir / f"{model_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Treinamento concluído. Melhor modelo: {results['best_model']['name']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {str(e)}", exc_info=True)
            raise
    
    # ==================== REGRESSÃO ====================
    
    def train_regression(
        self,
        df: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        top_n_models: int = 5
    ) -> Dict[str, Any]:
        """
        Treina múltiplos modelos de regressão automaticamente.
        """
        from pycaret.regression import (
            setup, compare_models, pull,
            predict_model, save_model, get_config
        )
        
        logger.info(f"Iniciando treinamento de regressão. Target: {target}")
        
        if feature_columns:
            columns_to_use = feature_columns + [target]
            df = df[columns_to_use]
        
        job_id = str(uuid.uuid4())[:8]
        
        try:
            # ===== SETUP =====
            logger.info("Executando setup do PyCaret...")
            
            reg_setup = setup(
                data=df,
                target=target,
                train_size=1 - test_size,
                session_id=42,
                verbose=False,
                html=False,
                
                normalize=True,
                normalize_method='zscore',
                handle_unknown_categorical=True,
                remove_multicollinearity=True,
                
                imputation_type='simple',
                numeric_imputation='mean',
                categorical_imputation='mode',
                
                n_jobs=-1,
            )
            
            # ===== COMPARAR MODELOS =====
            logger.info("Comparando modelos de regressão...")
            
            models_to_compare = [
                'rf',       # Random Forest
                'xgboost',  # XGBoost
                'lightgbm', # LightGBM
                'lr',       # Linear Regression
                'ridge',    # Ridge Regression
                'lasso',    # Lasso Regression
                'en',       # Elastic Net
                'dt',       # Decision Tree
                'knn',      # K-Nearest Neighbors
                'ada',      # AdaBoost
                'gbr',      # Gradient Boosting
                'et',       # Extra Trees
            ]
            
            best_model = compare_models(
                include=models_to_compare,
                n_select=top_n_models,
                sort='RMSE',
                verbose=False
            )
            
            results_df = pull()
            
            # ===== SALVAR MODELO =====
            model_filename = f"reg_{job_id}"
            model_path = self.models_dir / model_filename
            save_model(best_model, str(model_path))
            
            feature_names = get_config('X_train').columns.tolist()
            
            results = {
                "job_id": job_id,
                "problem_type": "regression",
                "target": target,
                "status": "completed",
                "created_at": datetime.now().isoformat(),
                
                "dataset_info": {
                    "total_rows": len(df),
                    "train_rows": int(len(df) * (1 - test_size)),
                    "test_rows": int(len(df) * test_size),
                    "n_features": len(feature_names),
                    "features": feature_names
                },
                
                "models_comparison": results_df.to_dict('records'),
                
                "best_model": {
                    "name": type(best_model).__name__,
                    "model_id": model_filename,
                    "path": str(model_path) + ".pkl",
                    "metrics": results_df.iloc[0].to_dict()
                },
                
                "ranking": [
                    {
                        "rank": i + 1,
                        "model": row['Model'],
                        "mae": row.get('MAE', 0),
                        "rmse": row.get('RMSE', 0),
                        "r2": row.get('R2', 0),
                        "mape": row.get('MAPE', 0)
                    }
                    for i, row in results_df.head(top_n_models).iterrows()
                ]
            }
            
            metadata_path = self.models_dir / f"{model_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Treinamento concluído. Melhor modelo: {results['best_model']['name']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {str(e)}", exc_info=True)
            raise
    
    # ==================== SÉRIES TEMPORAIS ====================
    
    def train_timeseries(
        self,
        df: pd.DataFrame,
        target: str,
        date_column: str,
        forecast_horizon: int = 30,
        top_n_models: int = 5
    ) -> Dict[str, Any]:
        """
        Treina múltiplos modelos de séries temporais.
        
        Nota: PyCaret Time Series ainda está em beta.
        Alternativa: usar statsmodels + prophet + sklearn
        """
        from pycaret.time_series import (
            setup, compare_models, pull,
            predict_model, save_model, get_config
        )
        
        logger.info(f"Iniciando treinamento de séries temporais. Target: {target}")
        
        job_id = str(uuid.uuid4())[:8]
        
        try:
            # Preparar dados para série temporal
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            df = df.set_index(date_column)
            
            # Garantir que o target é numérico
            ts_data = df[[target]].astype(float)
            
            # ===== SETUP =====
            logger.info("Executando setup do PyCaret Time Series...")
            
            ts_setup = setup(
                data=ts_data,
                target=target,
                fh=forecast_horizon,  # Horizonte de previsão
                session_id=42,
                verbose=False,
                html=False,
            )
            
            # ===== COMPARAR MODELOS =====
            logger.info("Comparando modelos de séries temporais...")
            
            best_model = compare_models(
                n_select=top_n_models,
                sort='MAPE',
                verbose=False
            )
            
            results_df = pull()
            
            # ===== SALVAR MODELO =====
            model_filename = f"ts_{job_id}"
            model_path = self.models_dir / model_filename
            save_model(best_model, str(model_path))
            
            results = {
                "job_id": job_id,
                "problem_type": "timeseries",
                "target": target,
                "date_column": date_column,
                "forecast_horizon": forecast_horizon,
                "status": "completed",
                "created_at": datetime.now().isoformat(),
                
                "dataset_info": {
                    "total_rows": len(df),
                    "date_range": {
                        "start": str(df.index.min()),
                        "end": str(df.index.max())
                    }
                },
                
                "models_comparison": results_df.to_dict('records'),
                
                "best_model": {
                    "name": type(best_model).__name__,
                    "model_id": model_filename,
                    "path": str(model_path) + ".pkl",
                    "metrics": results_df.iloc[0].to_dict()
                },
                
                "ranking": [
                    {
                        "rank": i + 1,
                        "model": row['Model'],
                        "mape": row.get('MAPE', 0),
                        "rmse": row.get('RMSE', 0),
                        "mae": row.get('MAE', 0)
                    }
                    for i, row in results_df.head(top_n_models).iterrows()
                ]
            }
            
            metadata_path = self.models_dir / f"{model_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no treinamento de séries temporais: {str(e)}", exc_info=True)
            raise
    
    # ==================== PREDIÇÃO ====================
    
    def predict(
        self,
        model_id: str,
        data: pd.DataFrame,
        problem_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Faz predições usando um modelo salvo.
        """
        if problem_type == "classification":
            from pycaret.classification import load_model, predict_model
        elif problem_type == "regression":
            from pycaret.regression import load_model, predict_model
        else:
            from pycaret.time_series import load_model, predict_model
        
        logger.info(f"Carregando modelo: {model_id}")
        
        model_path = self.models_dir / model_id
        model = load_model(str(model_path))
        
        logger.info(f"Fazendo predições para {len(data)} registros")
        
        predictions = predict_model(model, data=data)
        
        # Extrair coluna de predição
        if 'prediction_label' in predictions.columns:
            pred_col = 'prediction_label'
        elif 'Label' in predictions.columns:
            pred_col = 'Label'
        else:
            pred_col = predictions.columns[-1]
        
        result = {
            "model_id": model_id,
            "n_predictions": len(predictions),
            "predictions": predictions[pred_col].tolist()
        }
        
        # Adicionar probabilidades se disponível (classificação)
        if 'prediction_score' in predictions.columns:
            result["probabilities"] = predictions['prediction_score'].tolist()
        elif 'Score' in predictions.columns:
            result["probabilities"] = predictions['Score'].tolist()
        
        return result
    
    # ==================== UTILITÁRIOS ====================
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Lista todos os modelos salvos"""
        models = []
        
        for metadata_file in self.models_dir.glob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                models.append({
                    "model_id": metadata["best_model"]["model_id"],
                    "problem_type": metadata["problem_type"],
                    "target": metadata["target"],
                    "best_model_name": metadata["best_model"]["name"],
                    "created_at": metadata["created_at"]
                })
        
        return models
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Retorna informações detalhadas de um modelo"""
        metadata_path = self.models_dir / f"{model_id}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_id}")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)


################## Testando
# Instância global do serviço
automl_service = AutoMLService()

data = pd.read_csv("../data/classificacao/online_shoppers.csv")
result_df = automl_service.train_classification(data, target="Revenue")

breakpoint()
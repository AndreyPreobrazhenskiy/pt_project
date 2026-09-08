import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

import joblib
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ============ PYDANTIC МОДЕЛИ ============

class SimilarHost(BaseModel):
    host_id: str
    similarity_score: float
    cluster: int
    reason: str

class PredictionResponse(BaseModel):
    query_host: str
    query_cluster: int
    most_similar_hosts: List[SimilarHost]

class HostRequest(BaseModel):
    host_id: str = Field(..., description="ID компьютера, например, 'C14909'")

# ============ ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ============

model_instance = None
reference_df: Optional[pd.DataFrame] = None

# ============ LIFESPAN ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка моделей в память при старте приложения"""
    global model_instance, reference_df
    
    models_dir = Path("models")
    logging.info("🚀 Загрузка ML-моделей в память...")
    
    try:
        scaler = joblib.load(models_dir / "scaler.pkl")
        kmeans = joblib.load(models_dir / "kmeans.pkl")
        nn = joblib.load(models_dir / "nn.pkl")
        reference_df = pd.read_pickle(models_dir / "host_reference.pkl")
        
        class InferenceModel:
            def __init__(self, scaler, kmeans, nn, ref_df):
                self.scaler = scaler
                self.kmeans = kmeans
                self.nn = nn
                self.reference_df = ref_df
                self.host_ids = ref_df.index.to_numpy()
                self.feature_columns = [
                    'unique_dst_hosts', 'unique_dst_ports', 'total_bytes', 
                    'total_packets', 'avg_duration', 'tcp_ratio'
                ]

            def _generate_explanation(self, query_data, host_data):
                """Улучшенная логика объяснения со смысловой оценкой роли узла"""
                reasons = []
                
                # 1. Совпадение кластера
                if host_data['cluster'] == query_data['cluster']:
                    reasons.append(f"Находятся в одном кластере ({int(query_data['cluster'])}).")
                
                # 2. Смысловая оценка масштаба узла
                if query_data['unique_dst_hosts'] > 500 and host_data['unique_dst_hosts'] > 500:
                    reasons.append("Оба являются высоконагруженными инфраструктурными узлами (тысячи уникальных подключений).")
                elif query_data['unique_dst_hosts'] < 20 and host_data['unique_dst_hosts'] < 20:
                    reasons.append("Оба являются типичными клиентскими рабочими станциями с ограниченным кругом общения.")
                
                # 3. Оценка портов
                if query_data['unique_dst_ports'] > 10000 and host_data['unique_dst_ports'] > 10000:
                    reasons.append("Обслуживают огромное количество сетевых сервисов (>10 000 уникальных портов).")
                elif abs(query_data['unique_dst_ports'] - host_data['unique_dst_ports']) <= 10:
                    reasons.append(f"Используют схожее количество уникальных портов ({int(query_data['unique_dst_ports'])} vs {int(host_data['unique_dst_ports'])}).")
                
                # 4. Оценка протоколов
                if abs(query_data['tcp_ratio'] - host_data['tcp_ratio']) < 0.15:
                    reasons.append(f"Имеют схожую долю TCP-трафика (~{query_data['tcp_ratio']:.1%}).")
                
                # 5. Оценка длительности сессий
                if abs(query_data['avg_duration'] - host_data['avg_duration']) < 5:
                    reasons.append(f"Схожая средняя длительность сетевых сессий (~{query_data['avg_duration']:.1f} сек).")
                    
                if not reasons:
                    reasons.append("Демонстрируют схожие общие паттерны интенсивности трафика (байты/пакеты/длительность).")
                    
                return " ".join(reasons)

            def find_similar_and_explain(self, query_host_id: str, top_k: int = 5):
                if query_host_id not in self.reference_df.index:
                    raise ValueError(f"Хост {query_host_id} не найден")
                    
                query_features = self.reference_df.loc[[query_host_id]][self.feature_columns].to_numpy()
                query_scaled = self.scaler.transform(query_features)
                query_cluster = int(self.reference_df.loc[query_host_id, 'cluster'])
                
                distances, indices = self.nn.kneighbors(query_scaled)
                similar_indices = indices[0][1:top_k+1]
                similar_distances = distances[0][1:top_k+1]
                
                similar_hosts = []
                query_data = self.reference_df.loc[query_host_id]
                
                for idx, dist in zip(similar_indices, similar_distances):
                    host_id = self.host_ids[idx]
                    host_data = self.reference_df.iloc[idx]
                    
                    explanation = self._generate_explanation(query_data, host_data)
                        
                    similar_hosts.append(SimilarHost(
                        host_id=str(host_id),
                        similarity_score=round(1.0 - dist, 3),
                        cluster=int(host_data['cluster']),
                        reason=explanation
                    ))
                    
                return {
                    "query_host": str(query_host_id),
                    "query_cluster": query_cluster,
                    "most_similar_hosts": similar_hosts
                }

        model_instance = InferenceModel(scaler, kmeans, nn, reference_df)
        logging.info("✅ Модели успешно загружены!")
        
    except Exception as e:
        logging.error(f"❌ Ошибка загрузки моделей: {e}")
        raise
        
    yield
    
    logging.info("🛑 Завершение работы приложения, очистка памяти...")
    model_instance = None
    reference_df = None

# ============ FASTAPI ПРИЛОЖЕНИЕ ============

app = FastAPI(
    title="NTA Host Similarity Service",
    description="Сервис для поиска похожих узлов корпоративной сети по сетевому поведению",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "NTA Similarity Service is running. Check /docs for API documentation."}

@app.post("/api/v1/find_similar", response_model=PredictionResponse)
async def find_similar(request: HostRequest):
    """
    Поиск наиболее похожих узлов для заданного компьютера.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Модели еще не загружены или произошла ошибка при загрузке")
        
    try:
        result = model_instance.find_similar_and_explain(request.host_id, top_k=5)
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.get("/api/v1/stats")
async def get_stats():
    """Получить статистику по кластерам (для Web UI и аналитики)"""
    if reference_df is None:
        raise HTTPException(status_code=503, detail="Модели не загружены")
    
    cluster_counts = reference_df['cluster'].value_counts().sort_index().to_dict()
    return {
        "total_hosts": len(reference_df),
        "num_clusters": len(cluster_counts),
        "cluster_distribution": {str(k): int(v) for k, v in cluster_counts.items()}
    }

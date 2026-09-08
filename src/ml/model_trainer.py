import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

import polars as pl
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class HostBehaviorModel:
    def __init__(self, data_path: str, models_dir: str):
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        self.nn = NearestNeighbors(n_neighbors=6, metric='cosine')
        
        # признаки
        self.feature_columns = [
            'unique_dst_hosts', 'unique_dst_ports', 'total_bytes', 
            'total_packets', 'avg_duration', 'tcp_ratio'
        ]
        
        self.host_ids = None
        self.reference_df = None

    def train_and_save(self):
        logging.info(f"Загрузка данных из {self.data_path}...")
        df = pl.read_parquet(self.data_path)
        
        self.host_ids = df['src_computer'].to_numpy()
        self.reference_df = df.to_pandas().set_index('src_computer')
        
        X = df.select(self.feature_columns).to_numpy()
        
        logging.info("Масштабирование признаков (StandardScaler)...")
        X_scaled = self.scaler.fit_transform(X)
        
        logging.info("Обучение кластеризации (KMeans)...")
        clusters = self.kmeans.fit_predict(X_scaled)
        self.reference_df['cluster'] = clusters
        
        logging.info("Построение индекса ближайших соседей (NearestNeighbors)...")
        self.nn.fit(X_scaled)
        
        logging.info("Сохранение артефактов модели...")
        joblib.dump(self.scaler, self.models_dir / "scaler.pkl")
        joblib.dump(self.kmeans, self.models_dir / "kmeans.pkl")
        joblib.dump(self.nn, self.models_dir / "nn.pkl")
        self.reference_df.to_pickle(self.models_dir / "host_reference.pkl")
        
        logging.info("Обучение завершено!")
        self._print_cluster_summary()

    def _print_cluster_summary(self):
        print("\n Распределение хостов по кластерам:")
        cluster_counts = self.reference_df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            print(f"Кластер {cluster_id}: {count} хостов")
            
        print("\nСредние характеристики кластеров:")
        summary = self.reference_df.groupby('cluster')[self.feature_columns].mean()
        summary['unique_dst_hosts'] = summary['unique_dst_hosts'].round(1)
        summary['unique_dst_ports'] = summary['unique_dst_ports'].round(1)
        summary['tcp_ratio'] = summary['tcp_ratio'].round(3)
        print(summary)

    def _generate_explanation(self, query_data, host_data):
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
            return {"error": f"Хост {query_host_id} не найден в базе данных"}
            
        # получение признаков запрашиваемого хоста
        query_features = self.reference_df.loc[[query_host_id]][self.feature_columns].to_numpy()
        query_scaled = self.scaler.transform(query_features)
        query_cluster = int(self.reference_df.loc[query_host_id, 'cluster'])
        
        # поиск соседей
        distances, indices = self.nn.kneighbors(query_scaled)
        
        similar_indices = indices[0][1:top_k+1]
        similar_distances = distances[0][1:top_k+1]
        
        similar_hosts = []
        query_data = self.reference_df.loc[query_host_id]
        
        for idx, dist in zip(similar_indices, similar_distances):
            host_id = self.host_ids[idx]
            host_data = self.reference_df.iloc[idx]
            
            explanation = self._generate_explanation(query_data, host_data)
                
            similar_hosts.append({
                "host_id": str(host_id),
                "similarity_score": round(1.0 - dist, 3),
                "cluster": int(host_data['cluster']),
                "reason": explanation
            })
            
        return {
            "query_host": str(query_host_id),
            "query_cluster": query_cluster,
            "most_similar_hosts": similar_hosts
        }

if __name__ == "__main__":
    model = HostBehaviorModel(
        data_path="data/host_features.parquet",
        models_dir="models"
    )
    
    model.train_and_save()
    
    # тест поиска на реальном хосте из нашего вывода (например, C14909 - специфичный клиент)
    test_host = "C14909"
    print(f"\nТестовый запрос для хоста: {test_host}")
    result = model.find_similar_and_explain(test_host, top_k=4)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

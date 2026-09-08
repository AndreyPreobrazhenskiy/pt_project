import streamlit as st
import pandas as pd
import requests
from pathlib import Path

st.set_page_config(
    page_title="NTA - Поиск похожих хостов",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Network Traffic Analysis")
st.subheader("Поиск похожих узлов корпоративной сети")

st.sidebar.header("📊 Статистика")

models_dir = Path("models")
if (models_dir / "host_reference.pkl").exists():
    reference_df = pd.read_pickle(models_dir / "host_reference.pkl")
    
    cluster_counts = reference_df['cluster'].value_counts().sort_index()
    st.sidebar.write(f"**Всего хостов:** {len(reference_df)}")
    st.sidebar.write(f"**Кластеров:** {len(cluster_counts)}")
    
    st.sidebar.write("\n**Распределение по кластерам:**")
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(reference_df)) * 100
        st.sidebar.write(f"  • Кластер {cluster_id}: {count} ({percentage:.1f}%)")
else:
    reference_df = None
    st.sidebar.warning("Данные для статистики не найдены")

st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    host_id = st.text_input("🖥️ Введите ID компьютера:", placeholder="Например: C14909", value="C14909")
with col2:
    st.write("")
    st.write("")
    search_btn = st.button("🔎 Найти похожие", type="primary")

if search_btn and host_id:
    try:
        with st.spinner("Ищем похожие хосты..."):
            response = requests.post(
                "http://api:8000/api/v1/find_similar",
                json={"host_id": host_id},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
        
        if response.status_code == 200:
            result = response.json()
            
            st.success(f"✅ Найдено похожих хостов: {len(result['most_similar_hosts'])}")
            
            st.markdown("### 📋 Информация о хосте")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ID хоста", result['query_host'])
                st.metric("Кластер", result['query_cluster'])
            
            st.markdown("### 🎯 Похожие хосты")
            
            similar_df = pd.DataFrame(result['most_similar_hosts'])
            
            st.dataframe(
                similar_df,
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("### 💬 Объяснения похожести")
            for idx, row in similar_df.iterrows():
                with st.expander(f"🔗 {row['host_id']} (сходство: {row['similarity_score']:.3f}, кластер {row['cluster']})"):
                    st.write(row['reason'])
        
        elif response.status_code == 404:
            st.error(f"❌ Хост {host_id} не найден в базе данных")
        else:
            st.error(f"❌ Ошибка сервера: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Не удалось подключиться к API. Убедитесь, что сервис запущен и доступен по имени 'api'.")
    except requests.exceptions.Timeout:
        st.error("❌ Превышено время ожидания ответа от API.")
    except Exception as e:
        st.error(f"❌ Произошла ошибка: {str(e)}")

st.markdown("---")
st.markdown("""
### ℹ️ О проекте
Этот сервис анализирует сетевое поведение хостов корпоративной сети и находит похожие узлы.

**Как это работает:**
1. Для каждого хоста извлекаются признаки поведения (уникальные хосты, порты, объем трафика и т.д.)
2. Хосты кластеризуются с помощью KMeans
3. Для поиска похожих используется алгоритм Nearest Neighbors с косинусной метрикой
4. Генерируется объяснение, почему хосты считаются похожими

**Технологии:**
- Backend: FastAPI
- ML: scikit-learn (KMeans, NearestNeighbors)
- Frontend: Streamlit
- Контейнеризация: Docker
""")

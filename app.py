import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path

st.set_page_config(
    page_title="AI Data Analyst Agent by Maryzavr",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Data Analyst Agent by Maryzavr")
st.markdown("""
**Анализ данных с помощью AI-агента на базе GigaChat**

Загрузите CSV файл, задайте вопрос, и агент самостоятельно проведёт анализ.
""")

with st.sidebar:
    st.header("Информация")
    st.markdown("""
    ### Как пользоваться:
    1. Загрузите CSV файл
    2. Напишите вопрос
    3. Нажмите "Запустить анализ"

    ### Примеры:
    - "Посчитай сумму продаж"
    - "Найди среднее значение"
    - "Построй график"
    """)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Шаг 1: Загрузите данные")
    
    uploaded_file = st.file_uploader(
        "CSV файл",
        type=["csv"]
    )
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            st.session_state['csv_path'] = tmp_path

        df_preview = pd.read_csv(tmp_path)
        st.session_state['current_data'] = df_preview
        st.session_state['csv_name'] = uploaded_file.name
        
        st.success(f"Файл загружен: {uploaded_file.name}")
        st.write(f"**Размер:** {df_preview.shape[0]} строк × {df_preview.shape[1]} колонок")
        
        with st.expander("Предпросмотр"):
            st.dataframe(df_preview.head())

with col2:
    st.subheader("Шаг 2: Задайте вопрос")
    
    user_query = st.text_area(
        "Ваш вопрос:",
        placeholder="Пример: Проанализируй данные",
        height=100
    )
    
    custom_context = st.text_area(
        "Дополнительный контекст (по желанию):",
        placeholder=(
            f"Пример: Обрати внимание на колонку sales на целевую аудиторию,"
            f"на распределение по городам и т.д."
            ),
        height=80
        )
    
    analyze_button = st.button(
        "Запустить анализ",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_file or not user_query
    )

if analyze_button and uploaded_file and user_query:
    st.divider()
    st.subheader("Результаты анализа")

    from agent import run_analysis
    
    with st.spinner("🔄 Агент анализирует данные..."):
        try:

            result = run_analysis(
                df=st.session_state['current_data'],
                user_query=user_query,
                custom_context=custom_context if custom_context else None,
                dataset_name=st.session_state.get('csv_name', 'dataset')
            )

            st.markdown("### Отчет")
            st.markdown(result['analysis_report'])
            if result.get('chart_paths'):
              st.markdown("### 📊 Визуализации")
              for chart_path in result['chart_paths']:
                try:
                  with open(chart_path, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=500)

                except Exception as e:
                  st.warning(f"Не удалось загрузить график: {e}")             
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

            with st.expander("Подробности ошибки"):
                import traceback
                st.code(traceback.format_exc())

elif analyze_button and not uploaded_file:
    st.warning("Сначала загрузите CSV файл")
elif analyze_button and not user_query:
    st.warning("Введите вопрос для анализа")

st.divider()
st.caption("Powered by GigaChat + LangChain")

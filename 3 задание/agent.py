#Агент для анализа данных с использованием LangChain + GigaChat
import json
import os
import sys
import time
import glob
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from typing import List, Optional, Any, Dict, Union

import pandas as pd
import plotly.express as px
import streamlit as st

from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import BaseTool
from pydantic import Field

from google.colab import userdata

try:
    from gigachat import GigaChat as DirectGigaChat
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gigachat", "-q"])
    from gigachat import GigaChat as DirectGigaChat


FORBIDDEN_PATTERNS = [
    'ignore instructions', 'forget previous', 'system prompt',
    'игнорируй инструкции', 'забудь предыдущие', 'системный промпт',
    'взломай', 'обойди', 'смени роль', 'токен', 'пароль',
    'hack', 'bypass', 'token', 'password'
]


request_history = {}

def detect_prompt_injection(user_input: str, user_id: str = None) -> tuple[bool, str]:
    if not user_input or len(user_input.strip()) < 2:
        return True, "empty_request"
    if len(user_input) > 5000:
        return True, "too_long"
    if user_id:
        now = time.time()
        if user_id not in request_history:
            request_history[user_id] = []
        request_history[user_id] = [ts for ts in request_history[user_id] if ts > now - 60]
        if len(request_history[user_id]) >= 10:
            return True, "rate_limit"
        request_history[user_id].append(now)
    user_input_lower = user_input.lower()
    for word in FORBIDDEN_PATTERNS:
      if word in user_input_lower:
        return True, "suspicious_content"

    return False, "safe"


def retry_on_failure(max_retries=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Ошибка: {e}. Повтор через {delay} сек...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

try:
    from google.colab import userdata
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def get_api_key():
    api_key = os.environ.get('SBER_API_KEY')
    if not api_key:
        raise ValueError("API ключ не найден! Установите переменную окружения SBER_API_KEY")
    if not api_key:
        raise ValueError("""
        API ключ не найден! Добавьте его одним из способов:
        1. В Colab: нажмите на Secrets → добавить 'sber_api_key'
        2. В терминале: export SBER_API_KEY='ваш_ключ'
        3. В Streamlit Cloud: добавьте в .streamlit/secrets.toml
        """)
    return api_key


class GigaChatAdapter(BaseChatModel):
    temperature: float = Field(default=0.1)
    model: str = Field(default="GigaChat")
    scope: str = Field(default="GIGACHAT_API_PERS")
    verify_ssl_certs: bool = Field(default=False)
    max_tokens: int = Field(default=2000)
    
    _credentials: str = None
    _client: Any = None
    
    def __init__(self, credentials: str, **kwargs):
        init_params = {
            'temperature': kwargs.get('temperature', 0.1),
            'model': kwargs.get('model', 'GigaChat'),
            'scope': kwargs.get('scope', 'GIGACHAT_API_PERS'),
            'verify_ssl_certs': kwargs.get('verify_ssl_certs', False),
            'max_tokens': kwargs.get('max_tokens', 2000)
        }
        super().__init__(**init_params)
        self._credentials = credentials
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            self._client = DirectGigaChat(
                credentials=self._credentials,
                verify_ssl_certs=self.verify_ssl_certs,
                scope=self.scope,
                model=self.model
            )
        return self._client
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs):
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = "assistant"
            api_messages.append({"role": role, "content": msg.content})
        
        client = self._get_client()
        response = client.chat(
            payload={
                "messages": api_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )
        
        content = response.choices[0].message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
    
    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs):
        return self._generate(messages, stop, **kwargs)
    
    def bind_tools(self, tools: List[Union[BaseTool, Dict]], **kwargs):
        return self
    
    @property
    def _llm_type(self) -> str:
        return f"gigachat-{self.model}"

def init_gigachat():
    """Инициализация GigaChat через адаптер"""
    api_key = get_api_key()
    return GigaChatAdapter(
        credentials=api_key,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat",
        temperature=0.1,
        max_tokens=2000
    )


def load_current_data() -> pd.DataFrame:
    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        raise ValueError("Данные не загружены. Сначала загрузите файл.")
    if isinstance(st.session_state['current_data'], pd.DataFrame):
        return st.session_state['current_data']
    if isinstance(st.session_state['current_data'], str):
        if not os.path.exists(st.session_state['current_data']):
            raise FileNotFoundError(f"Файл не найден: {st.session_state['current_data']}")
        df = pd.read_csv(st.session_state['current_data'])
        st.session_state['current_data'] = df
        return df
    raise ValueError("Некорректный формат данных")


@tool
def get_column_list() -> str:
    """Получить список всех колонок в датасете."""
    try:
        df = load_current_data()
        columns = df.columns.tolist()
        return (f"Колонки в датасете ({len(columns)} шт):\n" +
                "\n".join(f"  • {col}" for col in columns))
    except FileNotFoundError:
        return f"Файл не найден"
    except Exception as e:
        return f"Ошибка: {str(e)}"

@tool
def python_execution_tool(code: str) -> str:
    """Выполнить Python код для расчетов, статистики и обработки данных."""
    dangerous = ['__import__', 'eval(', 'exec(', 'open(', 'os.', 'subprocess']
    for cmd in dangerous:
       if cmd in code:
         return f"Заблокировано: {cmd}"
    df = load_current_data()
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        safe_vars = {
            'df': df,
            'pd': pd,
            'np': np,
            'print': print
        }
        exec(code, {"__builtins__": {}}, safe_vars)
        output = sys.stdout.getvalue()
        if len(output) > 3000:
            output = output[:3000] + "\n... (вывод обрезан)"
        return f"Результат:\n{output}"
    except Exception as e:
        return f"Ошибка выполнения: {str(e)}"
    finally:
        sys.stdout = old_stdout


@tool
def graph_generator(code):
    """Создать график через plotly"""
    try:
        df = load_current_data()
    except:
        return "Ошибка загрузки данных"
    try:
        globals_dict = {
            'df': df,
            'pd': pd,
            'px': px,
            'plt': plt,
            'np': np
        }
        exec(code, {"__builtins__": {}}, globals_dict)
        fig = globals_dict.get('fig')
        if fig is None:
          return "Не найден график. Создайте переменную 'fig'"
        chart_name = f"chart_{int(time.time())}.html"
        chart_path = os.path.join(tempfile.gettempdir(), chart_name)
        fig.write_html(chart_path)
        return f"График сохранен: {chart_name}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool
def get_basic_stats() -> str:
    """Получить базовую статистику по датасету."""
    try:
        df = load_current_data()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=['object', 'category']
            ).columns.tolist()
        datetime_cols = df.select_dtypes(
            include=['datetime64']
            ).columns.tolist()
        stats = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "column_types": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "datetime": len(datetime_cols)
            },
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        }
        numeric_stats = {}
        for col in numeric_cols:
            if not df[col].isnull().all():
                numeric_stats[col] = {
                    "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                    "median": float(df[col].median()) if pd.notna(df[col].median()) else None,
                    "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
                    "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                    "max": float(df[col].max()) if pd.notna(df[col].max()) else None
                }
        stats["numeric_stats"] = numeric_stats
        return json.dumps(stats, ensure_ascii=False, indent=2, default=str)
    except FileNotFoundError:
        return f"Файл не найден"
    except pd.errors.EmptyDataError:
        return "Файл пуст или не содержит данных"
    except Exception as e:
        import traceback
        print(f"Ошибка в get_basic_stats: {traceback.format_exc()}")
        return f"Ошибка: {str(e)}"


@retry_on_failure(max_retries=3, delay=2)
def run_analysis(
    df: pd.DataFrame,
    user_query: str,
    custom_context: str=None,
    user_id: str=None,
    dataset_name: str = "dataset"
) -> dict:
    """
    Запускает агента для анализа данных.
    Возвращает результаты в структурированном виде.
    """
    print(f"Запуск анализа для файла")
    print(f"Запрос: {user_query[:200]}...")

    if df is None:
        raise ValueError("DataFrame не передан")

    is_dangerous, reason = detect_prompt_injection(user_query, user_id)
    if is_dangerous:
        print(f"Обнаружена попытка prompt injection! Причина: {reason}")
        with open("security_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {reason} - {user_query[:100]}\n")
        return {
            "analysis_report": (f"##Запрос заблокирован\n"
                                f"**Причина:** {reason}\n"
                                f"Ваш запрос содержит потенциально опасный контент."
                                ),
            "metrics": ["Безопасность: запрос отклонён"],
            "key_insights": [f"Обнаружена угроза: {reason}"],
            "chart_paths": [],
            "conclusion": f"Запрос заблокирован ({reason})",
            "dataset_name": dataset_name
        }
    llm = init_gigachat()
    tools = [get_column_list, get_basic_stats, python_execution_tool, graph_generator]
    system_prompt = f"""Ты — эксперт по анализу данных.
Твоя задача - самостоятельно провести анализ данных.
Данные: {len(df)} строк, {len(df.columns)} колонок.
Колонки: {', '.join(df.columns[:15])}
Пример данных:
{df.head(5).to_string()}
У тебя есть доступ к этим инструментам:
- get_column_list: узнать список и типы колонок
- get_basic_stats: получить статистику и информацию о данных
- python_execution_tool: выполнить Python код для расчетов
- graph_generator: создать интерактивные графики

Правила работы:
1. Сначала изучи структуру данных (get_column_list)
2. Затем получи базовую статистику (get_basic_stats)
3. По необходимости используй python_execution_tool для сложных расчетов
4. Создавай графики для визуализации ключевых закономерностей
5. Ответ должен быть на русском языке
6. Используй Markdown для форматирования
ВАЖНОЕ ПРАВИЛО:
- НЕ ПИШИ graph_generator(...) или python_execution_tool(...) в тексте ответа
- Используй доступные инструменты через их вызов (это происходит автоматически)
- Твой ответ должен содержать ТОЛЬКО текстовый отчет
- Графики и расчеты создаются автоматически через инструменты
"""

    if custom_context:
        system_prompt += f"""
ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ОТ ПОЛЬЗОВАТЕЛЯ:
{custom_context}
Пожалуйста, учти этот контекст при анализе данных.
"""
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SystemMessage(content=system_prompt)
    )

    full_query = f"""Проанализируй данные согласно запросу пользователя: {user_query}
Пожалуйста, проведи анализ и предоставь структурированный отчет с выводами."""
    try:
      result = agent.invoke({
            "messages": [HumanMessage(content=full_query)]
        })
      messages = result.get("messages", [])
      output_text = messages[-1].content if messages else str(result)
      chart_paths = glob.glob(os.path.join(tempfile.gettempdir(), "chart_*.html"))
      return {
          "analysis_report": output_text,
          "metrics": ["Анализ завершён успешно"],
          "key_insights": ["Результаты представлены в отчёте"],
          "chart_paths": chart_paths,
          "conclusion": output_text[:300] if len(output_text) > 300 else output_text,
          "dataset_name": dataset_name
          }
    except Exception as e:
        import traceback
        print(f"Ошибка в run_analysis: {traceback.format_exc()}")
        return {
            "analysis_report": (
                f"##Ошибка выполнения анализа\n```\n{str(e)}\n"
                f"Пожалуйста, попробуйте упростить запрос."
                ),
            "metrics": [],
            "key_insights": [f"Ошибка: {str(e)[:100]}"],
            "chart_paths": [],
            "conclusion": f"Ошибка: {str(e)[:100]}",
            "dataset_name": dataset_name
        }

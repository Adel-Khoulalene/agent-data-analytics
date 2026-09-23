import os
import streamlit as st
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from typing import Annotated, TypedDict
from pydantic import BaseModel
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

        
# ------------------------------------------------------------------
# 1. Configuration de la page Streamlit
# ------------------------------------------------------------------
st.set_page_config(page_title="Agent Data Science", page_icon="📊", layout="wide")
st.title("📊 Agent Autonome d'Analyse de Données")
st.caption("LangGraph + Groq (Qwen 2.5 Coder) + DuckDB + Matplotlib (Zero Code Execution)")

# Récupération de la clé API
# ------------------------------------------------------------------
# 2. Récupération DE LA CLÉ (Doit être faite AVANT TOUT LE RESTE)
# ------------------------------------------------------------------
groq_api_key = None

if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
elif "GROQ_API_KEY" in os.environ:
    groq_api_key = os.environ["GROQ_API_KEY"]

if not groq_api_key:
    st.error("🔑 Clé API Groq introuvable. Allez dans Settings > Secrets et ajoutez : GROQ_API_KEY = \"gsk_...\"")
    st.stop()

# ------------------------------------------------------------------
# 3. Bouton de test (Placé APRÈS la définition de groq_api_key)
# ------------------------------------------------------------------
if st.sidebar.button("Tester la connexion Groq"):
    try:
        test_llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
        res = test_llm.invoke("Dis 'Connexion réussie !'")
        st.sidebar.success(res.content)
    except Exception as e:
        st.sidebar.error(f"Erreur Groq : {e}")
        
# ------------------------------------------------------------------
# 2. Base de données DuckDB
# ------------------------------------------------------------------
@st.cache_resource
def init_db():
    conn = duckdb.connect(database=':memory:', read_only=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clients (
            id INT PRIMARY KEY,
            nom VARCHAR,
            total_depense DOUBLE
        );
        DELETE FROM clients;
        INSERT INTO clients VALUES 
        (1, 'Alice', 1200.50),
        (2, 'Bob', 850.00),
        (3, 'Charlie', 2300.10),
        (4, 'Diana', 3100.00),
        (5, 'Aurel', 450.75);
    """)
    return conn

conn = init_db()

DB_SCHEMA = """
Table: clients
Colonnes:
- id (INTEGER)
- nom (VARCHAR)
- total_depense (DOUBLE)
"""

# ------------------------------------------------------------------
# 3. Structure des données et État LangGraph
# ------------------------------------------------------------------
class ChartConfig(BaseModel):
    chart_type: Literal["bar", "line", "scatter"]
    x_column: str
    y_column: str
    title: str

class SQLState(TypedDict):
    messages: Annotated[list, add_messages]
    sql_query: str
    sql_result: str
    error: str
    retry_count: int
    chart_config: str
    chart_path: str

# ------------------------------------------------------------------
# 4. Agent LangGraph
# ------------------------------------------------------------------

def build_agent(api_key: str):
    # ✅ Remplace par un modèle officiel supporté par Groq (ex: Llama 3.3 70B) :
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0
    )

    def generate_sql(state: SQLState):
        retry_count = state.get("retry_count", 0)
        error = state.get("error", "")
        prompt = f"Tu es un expert SQL DuckDB.\nSchéma:\n{DB_SCHEMA}\nGénère UNIQUEMENT la requête SQL sans markdown."
        if error:
            prompt += f"\nErreur précédente : '{error}'. Corrige-la."
        else:
            prompt += f"\nDemande : {state['messages'][-1].content}"
            
        response = llm.invoke(prompt)
        clean_sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
        return {"sql_query": clean_sql, "retry_count": retry_count + 1}

    def execute_sql(state: SQLState):
        query = state["sql_query"]
        try:
            res = conn.execute(query).fetchall()
            return {"sql_result": str(res), "error": ""}
        except Exception as e:
            return {"sql_result": "", "error": str(e)}

    def generate_chart(state: SQLState):
        sql_result = state.get("sql_result", "")
        user_query = state['messages'][-1].content
        parser = JsonOutputParser(pydantic_object=ChartConfig)
        prompt = f"Données DuckDB : {sql_result}\nDemande : {user_query}\nExtrais la config JSON:\n{parser.get_format_instructions()}"
        
        config_dict = {}
        try:
            response = llm.invoke(prompt)
            config_dict = parser.parse(response.content)
            config = ChartConfig(**config_dict)
            
            raw_data = eval(sql_result)
            df = pd.DataFrame(raw_data)
            if len(df.columns) >= 2:
                df.columns = [config.x_column, config.y_column]
            
            plt.clf()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            if config.chart_type == "bar":
                ax.bar(df[config.x_column].astype(str), df[config.y_column], color="#1f77b4")
            elif config.chart_type == "line":
                ax.plot(df[config.x_column].astype(str), df[config.y_column], marker='o', color="#ff7f0e")
            elif config.chart_type == "scatter":
                ax.scatter(df[config.x_column].astype(str), df[config.y_column], color="#2ca02c")
                
            ax.set_title(config.title)
            ax.set_xlabel(config.x_column)
            ax.set_ylabel(config.y_column)
            plt.tight_layout()
            
            chart_path = "chart.png"
            plt.savefig(chart_path)
            plt.close()
        except Exception:
            chart_path = ""
            
        return {"chart_config": str(config_dict), "chart_path": chart_path}

    def should_retry(state: SQLState):
        if not state.get("error"):
            return "success"
        elif state.get("retry_count", 0) >= 3:
            return "max_retries"
        else:
            return "retry"

    builder = StateGraph(SQLState)
    builder.add_node("generate_sql", generate_sql)
    builder.add_node("execute_sql", execute_sql)
    builder.add_node("generate_chart", generate_chart)

    builder.add_edge(START, "generate_sql")
    builder.add_edge("generate_sql", "execute_sql")
    builder.add_conditional_edges("execute_sql", should_retry, {
        "success": "generate_chart",
        "max_retries": END,
        "retry": "generate_sql"
    })
    builder.add_edge("generate_chart", END)

    return builder.compile()

agent = build_agent(groq_api_key)

# ------------------------------------------------------------------
# 5. Interface Chat Streamlit
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sql" in msg and msg["sql"]:
            st.code(msg["sql"], language="sql")
        if "chart_path" in msg and msg["chart_path"]:
            st.image(msg["chart_path"])

user_input = st.chat_input("Ex: Affiche le total des dépenses des clients sous forme de graphique...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyse et génération en cours..."):
            initial_state = {"messages": [("user", user_input)], "retry_count": 0, "error": ""}
            final_state = agent.invoke(initial_state)

            sql_query = final_state.get("sql_query", "")
            sql_result = final_state.get("sql_result", "")
            chart_path = final_state.get("chart_path", "")

            st.write("Résultat de l'analyse :")
            if sql_query:
                st.markdown("**Requête SQL exécutée :**")
                st.code(sql_query, language="sql")
            
            if sql_result:
                st.markdown("**Données retournées :**")
                try:
                    df = pd.DataFrame(eval(sql_result))
                    st.dataframe(df, use_container_width=True)
                except Exception:
                    st.write(sql_result)

            if chart_path:
                st.markdown("**Visualisation :**")
                st.image(chart_path)

            st.session_state.messages.append({
                "role": "assistant",
                "content": "Analyse complétée.",
                "sql": sql_query,
                "chart_path": chart_path
            })
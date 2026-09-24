import os
import datetime

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
st.title("📊 Agent Autonome d'Analyse de Données Multi-CSV")
st.caption("LangGraph + Groq (openai/gpt-oss-120b) + DuckDB (Support CSV dynamique)")

# ------------------------------------------------------------------
# 2. Récupération de la clé API Groq
# ------------------------------------------------------------------
groq_api_key = None

if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
elif "GROQ_API_KEY" in os.environ:
    groq_api_key = os.environ["GROQ_API_KEY"]

if not groq_api_key:
    st.error("🔑 Clé API Groq introuvable. Ajoutez `GROQ_API_KEY` dans vos Secrets Streamlit.")
    st.stop()

# ------------------------------------------------------------------
# 3. Sidebar : Test de connexion & File Uploader CSV
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration & Données")
    
    if st.button("Tester la connexion Groq", key="btn_test_groq"):
        try:
            test_llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)
            res = test_llm.invoke("Dis 'Connexion réussie !'")
            st.success(res.content)
        except Exception as e:
            st.error(f"Erreur Groq : {e}")

    st.markdown("---")
    st.subheader("📁 Importer des données")
    uploaded_file = st.file_uploader(
    "Téléversez un fichier (CSV, TXT, DATA)", 
    type=["csv", "txt", "data", "log"]
    )

# ------------------------------------------------------------------
# 4. Gestion de la Base de Données DuckDB & Schéma Dynamique
# ------------------------------------------------------------------
@st.cache_resource
def get_db_connection():
    return duckdb.connect(database=':memory:', read_only=False)

conn = get_db_connection()

if uploaded_file is not None:
    try:
        # 1. Lecture du fichier avec détection des séparateurs
        df_uploaded = pd.read_csv(
            uploaded_file,
            sep=r'[;,|\t]+',
            engine='python'
        )

        # 2. Nettoyage des guillemets et espaces
        df_uploaded = df_uploaded.apply(
            lambda col: col.astype(str)
            .str.replace('"', '', regex=False)
            .str.replace("'", "", regex=False)
            .str.strip()
        )

        df_uploaded.columns = [
            col.replace('"', '').replace("'", "").strip()
            for col in df_uploaded.columns
        ]

        # 3. Suppression des colonnes vides
        df_uploaded = df_uploaded.loc[
            :, ~df_uploaded.columns.str.contains('^Unnamed')
        ]

        # 4. Conversion automatique des types
        for col in df_uploaded.columns:

            col_str = (
                df_uploaded[col]
                .astype(str)
                .str.strip()
                .str.replace('"', '', regex=False)
                .str.replace("'", "", regex=False)
            )

            non_empty = col_str[
                ~col_str.isin(
                    ["NA", "nan", "NaN", "<NA>", "", "None"]
                )
            ]

            # A. Dates YYYYMMDD
            if (
                len(non_empty) > 0
                and non_empty.str.fullmatch(r"(19|20)\d{6}").all()
            ):
                converted_date = pd.to_datetime(
                    col_str,
                    format="%Y%m%d",
                    errors="coerce"
                )

                if converted_date.notna().sum() > 0:
                    df_uploaded[col] = converted_date
                    continue

            # B. Dates classiques
            try:
                converted_date = pd.to_datetime(
                    col_str,
                    format="mixed",
                    errors="coerce",
                    dayfirst=True
                )

                if (
                    len(non_empty) > 0
                    and converted_date.notna().sum() / len(non_empty) >= 0.5
                ):
                    df_uploaded[col] = converted_date
                    continue

            except Exception:
                pass

            # C. Numérique
            converted_num = pd.to_numeric(
                col_str.str.replace(",", ".", regex=False),
                errors="coerce"
            )

            if (
                converted_num.notna().sum() > 0
                and not pd.api.types.is_datetime64_any_dtype(
                    df_uploaded[col]
                )
            ):
                df_uploaded[col] = converted_num

        # 5. Injection dans DuckDB
        table_name = "dataset"

        conn.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            "SELECT * FROM df_uploaded"
        )

        # 6. Inspection dynamique du schéma
        schema_info = conn.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()

        cols_str = "\n".join(
            [f"- {col[1]} ({col[2]})" for col in schema_info]
        )

        db_schema = f"Table: {table_name}\nColonnes:\n{cols_str}"

        st.sidebar.success(
            f"Fichier `{uploaded_file.name}` chargé "
            f"({len(df_uploaded)} lignes) !"
        )

        st.sidebar.dataframe(
            df_uploaded.head(3),
            use_container_width=True
        )

    except Exception as e:
        st.sidebar.error(
            f"Erreur lors du chargement du fichier : {e}"
        )
        st.stop()

else:
    # Table d'exemple par défaut
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

    db_schema = """
    Table: clients
    Colonnes:
    - id (INTEGER)
    - nom (VARCHAR)
    - total_depense (DOUBLE)
    """

    st.sidebar.info(
        "💡 Aucun fichier fourni : utilisation de la table "
        "par défaut `clients`."
    )

# ------------------------------------------------------------------
# 5. Structure Pydantic & État LangGraph
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
# 6. Construction de l'Agent Autonome
# ------------------------------------------------------------------
def build_agent(api_key: str, schema_context: str):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=0
    )

    def generate_sql(state: SQLState):
        retry_count = state.get("retry_count", 0)
        error = state.get("error", "")
        prompt = f"Tu es un expert SQL DuckDB.\nSchéma disponible :\n{schema_context}\nGénère UNIQUEMENT la requête SQL adaptée sans balises markdown."
        if error:
            prompt += f"\nL'exécution précédente a échoué avec l'erreur : '{error}'. Corrige la requête."
        else:
            prompt += f"\nDemande de l'utilisateur : {state['messages'][-1].content}"
            
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
        prompt = f"Résultat SQL : {sql_result}\nDemande utilisateur : {user_query}\nFormate la configuration de visualisation au format JSON selon ces instructions :\n{parser.get_format_instructions()}"
        
        config_dict = {}
        chart_path = ""
        try:
            response = llm.invoke(prompt)
            config_dict = parser.parse(response.content)
            config = ChartConfig(**config_dict)
            
            raw_data = eval(sql_result)
            df = pd.DataFrame(raw_data)

            # Vérification des colonnes demandées
            if config.x_column in df.columns and config.y_column in df.columns:
                df = df[[config.x_column, config.y_column]].copy()
            else:
                if len(df.columns) >= 2:
                    df = df.iloc[:, :2].copy()
                    df.columns = [config.x_column, config.y_column]
                else:
                    raise ValueError("Le résultat SQL ne contient pas assez de colonnes pour générer le graphique.")

            # Nettoyage de l'axe X
            df[config.x_column] = df[config.x_column].astype(str)

            # Conversion explicite de l'axe Y en numérique
            df[config.y_column] = pd.to_numeric(
                df[config.y_column],
                errors="coerce"
            )

            # Suppression des lignes invalides
            df = df.dropna(subset=[config.y_column])

            if df.empty:
                raise ValueError(
                    f"Aucune valeur numérique exploitable dans la colonne '{config.y_column}'."
                )

            # Génération du graphique
            plt.close("all")

            fig, ax = plt.subplots(figsize=(10, 5))

            if config.chart_type == "bar":
                ax.bar(
                    df[config.x_column],
                    df[config.y_column]
                )

            elif config.chart_type == "line":
                ax.plot(
                    df[config.x_column],
                    df[config.y_column],
                    marker="o"
                )

            elif config.chart_type == "scatter":
                ax.scatter(
                    df[config.x_column],
                    df[config.y_column]
                )

            ax.set_title(str(config.title))
            ax.set_xlabel(str(config.x_column))
            ax.set_ylabel(str(config.y_column))

            plt.xticks(rotation=45)
            plt.tight_layout()

            chart_path = "chart.png"
            fig.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique : {e}")
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

# Instanciation dynamique avec le schéma courant
agent = build_agent(groq_api_key, db_schema)

# ------------------------------------------------------------------
# 7. Interface Chat Streamlit
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

user_input = st.chat_input("Posez une question sur vos données (ex: Donnes-moi la répartition par catégorie)...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyse et génération en cours via DuckDB + Groq..."):
            initial_state = {"messages": [("user", user_input)], "retry_count": 0, "error": ""}
            final_state = agent.invoke(initial_state)

            sql_query = final_state.get("sql_query", "")
            sql_result = final_state.get("sql_result", "")
            chart_path = final_state.get("chart_path", "")

            st.write("Résultat de l'analyse :")
            if sql_query:
                st.markdown("**Requête SQL générée :**")
                st.code(sql_query, language="sql")
            
            if sql_result:
                st.markdown("**Données extraites :**")
                try:
                    df_res = pd.DataFrame(eval(sql_result))
                    st.dataframe(df_res, use_container_width=True)
                except Exception:
                    st.write(sql_result)

            if chart_path:
                st.markdown("**Graphique généré :**")
                st.image(chart_path)

            st.session_state.messages.append({
                "role": "assistant",
                "content": "Analyse terminée avec succès.",
                "sql": sql_query,
                "chart_path": chart_path
            })
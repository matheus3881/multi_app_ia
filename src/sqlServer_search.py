import pyodbc
import json
import time
from sentence_transformers import SentenceTransformer
import streamlit as st

def connect_to_sqlserver():
    """Conecta ao SQL Server."""
    # É uma boa prática carregar segredos de forma segura,
    # mas para este exemplo, a connection string está aqui.
    conn_str = (
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        r'SERVER=devapprhel02;'
        r'DATABASE=IA;'
        r'UID=sa;'
        r'PWD=yourStrong(!)Password;'
        r'Encrypt=yes;'
        r'TrustServerCertificate=yes;'
    )
    return pyodbc.connect(conn_str)

@st.cache_resource
def load_sentence_transformer():
    """Carrega o modelo de embeddings e o armazena em cache."""
    st.write("Carregando modelo 'BAAI/bge-m3' (isso acontece apenas uma vez)...")
    start_time = time.perf_counter()
    model = SentenceTransformer('BAAI/bge-m3', device='cpu')
    elapsed = time.perf_counter() - start_time
    st.success(f"Modelo carregado em {elapsed:.2f} segundos.")
    return model

def display_results(results, query_text):
    """Exibe os resultados da busca puramente vetorial no Streamlit."""
    st.divider()
    st.subheader(f"Resultados da Busca Vetorial") # <--- MODIFICADO
    st.markdown(f"**Consulta**: **'{query_text}'**")
    st.markdown(f"**Total de resultados**: **{len(results)}**")

    for i, result in enumerate(results, 1):
        st.divider()
        st.markdown(f"#### Resultado {i}")
        # <--- MODIFICADO: Removido 'Score Combinado' e 'Edit Distance'
        st.markdown(f"**Similaridade Vetorial**: **{result['vector_similarity']:.4f}** (Document ID: {result['document_id']})")
        
        with st.expander("**Ver Resumo e Documento**"):
            st.markdown(f"**Resumo:**")
            st.write(f"{result['summary']}")
            st.markdown(f"**Documento (primeiros 300 caracteres):**")
            st.write(f"{result['document'][:300]}...")

    st.divider()

def search_by_sql_vector_inline(query_text, model, top_k=5):
    """Executa uma busca puramente vetorial no SQL Server."""
    
    embed_start = time.perf_counter()
    query_embedding = model.encode([query_text])[0].tolist()
    embed_time = time.perf_counter() - embed_start

    # Prepara os literais para injeção na query.
    # ATENÇÃO: Para produção, use parâmetros para evitar SQL Injection.
    query_vector_json = json.dumps(query_embedding).replace("'", "''")

    conn = connect_to_sqlserver()
    cursor = conn.cursor()

    try:
        # --- QUERY SIMPLIFICADA PARA BUSCA PURAMENTE VETORIAL ---
        sql = f"""
        SELECT TOP ({top_k})
            d.DocumentId,
            d.Document,
            d.Summary,
            (1 - VECTOR_DISTANCE('cosine', e.EmbeddingData, CAST(CAST(N'{query_vector_json}' AS NVARCHAR(MAX)) AS VECTOR(1024)))) AS VectorSimilarity
        FROM dbo.EmbeddingsFaleConosco e
        INNER JOIN dbo.DocumentsFaleConosco d ON e.DocumentId = d.DocumentId
        ORDER BY VectorSimilarity DESC;
        """

        sql_start = time.perf_counter()
        cursor.execute(sql)
        rows = cursor.fetchall()
        sql_time = time.perf_counter() - sql_start

        # Imprime os tempos no console (para debug)
        print(f"Tempos (s): embed={embed_time:.3f}, sql_exec={sql_time:.3f}")

        formatted_results = []
        for row in rows:
            formatted_results.append({
                'document_id': row.DocumentId,
                'document': row.Document,
                'summary': row.Summary,
                'vector_similarity': row.VectorSimilarity
                # <--- MODIFICADO: Lógica de score combinado removida
            })

        return formatted_results

    except Exception as e:
        st.error(f"Erro na busca SQL: {e}") # Exibe o erro na interface do Streamlit
        print(f"Erro na busca inline (SQL): {e}") # Imprime o erro no console
        return []
    
    finally:
        # Garante que a conexão seja sempre fechada
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
            


def buscar_no_banco(query, top_k):
    model_object = load_sentence_transformer()
    resullts = search_by_sql_vector_inline(query_text=query, top_k=top_k, model=model_object)
    if resullts:
        display = display_results(resullts, query)
        return display
    else:
        st.info("Nenhu resultado encotrado para a sua busca")



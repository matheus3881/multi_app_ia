# -*- coding: utf-8 -*-

# --- Importações Essenciais ---
import traceback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import io
import os
import re
import easyocr
import faiss
import numpy as np
import pandas as pd
import streamlit as st
import pymupdf  # Fitz
import torch
import cv2
from PIL import Image

# --- Tesseract ---
import pytesseract
from pytesseract import Output
# Configure o caminho para o executável do Tesseract, se necessário
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\msantos\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# --- hugginf face ---
# --- Machine Learning & LangChain ---


# Configuração para evitar problemas de duplicidade de libs
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# reader = easyocr.Reader(['pt'], gpu=False)  # idioma português


# ==============================================================================
# ETAPA 1: FUNÇÕES CACHEADAS (Otimização de Performance)
# ==============================================================================

@st.cache_resource
def carregar_modelo_embedding():
    """
    Carrega o modelo de embedding SentenceTransformer.
    Graças ao @st.cache_resource, isso só acontece UMA VEZ quando o app inicia.
    """
    print("--- CARREGANDO O MODELO DE EMBEDDING (ISSO SÓ DEVE APARECER UMA VEZ) ---")
    return SentenceTransformer('BAAI/bge-m3', device='cpu')


@st.cache_data
def processar_documento_cacheado(file_bytes, file_name):
    """
    Função "mestra" que executa todo o processamento pesado (OCR, Embeddings).
    Graças ao @st.cache_data, só será executada se o conteúdo do arquivo for novo.
    """
    print(
        f"\n--- EXECUTANDO PROCESSAMENTO PESADO (CACHE MISS) PARA: {file_name} ---")

    # 1. Converter arquivo (PDF/imagem) em uma lista de imagens PIL
    imagens = processar_arquivo(file_bytes, file_name)

    # 2. Realizar OCR nas imagens para extrair texto e coordenadas
    documentos_ocr = realizar_ocr(imagens)

    # 3. Dividir o texto em chunks menores, se necessário
    chunks = dividir_em_chunks(documentos_ocr)

    if not chunks:
        print("--- AVISO: Nenhum chunk de texto foi gerado após OCR e divisão. ---")
        return None, None  # Retorna None se não houver texto

    # 4. Criar o índice vetorial FAISS a partir dos chunks
    index, chunks_final = criar_faiss(chunks)

    return index, chunks_final

# ==============================================================================
# ETAPA 2: FUNÇÕES DE PROCESSAMENTO DE ARQUIVO E OCR
# ==============================================================================


def processar_arquivo(arquivo_bytes, nome):
    """Verifica o tipo do arquivo e o converte para uma lista de imagens PIL."""
    if nome.lower().endswith(".pdf"):
        return processar_pdf(arquivo_bytes)
    elif nome.lower().endswith((".png", ".jpg", ".jpeg")):
        return [Image.open(io.BytesIO(arquivo_bytes)).convert("RGB")]
    else:
        raise ValueError(
            "Formato de arquivo não suportado. Envie um PDF ou PNG.")


def processar_pdf(pdf_bytes):
    """Converte cada página de um PDF para uma imagem PIL de alta resolução."""
    imagens = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for pagina in doc:
            pix = pagina.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            imagens.append(img)
    return imagens


def realizar_ocr(imagens: list):
    """Realiza OCR e retorna Document com bbox preciso por palavra."""
    documentos_completos = []

    for i, img in enumerate(imagens):
        # Converte a imagem PIL para o formato CV (Numpy array)
        img_cv = np.array(img)

        # Usa Output.DICT para obter um dicionário
        data = pytesseract.image_to_data(img_cv, output_type=Output.DICT)

        n_boxes = len(data['text'])

        # Variáveis para agrupar as palavras em linhas
        linhas_agrupadas = {}

        for j in range(n_boxes):
            confianca = int(data['conf'][j])
            texto_palavra = data['text'][j]

            # Filtra palavras com baixa confiança ou vazias
            if confianca > 60 and texto_palavra.strip():
                # Cria uma chave única para cada linha (página, bloco, parágrafo, linha)
                chave_linha = (
                    i + 1,  # Número da página (começando em 1)
                    data['block_num'][j],
                    data['par_num'][j],
                    data['line_num'][j]
                )

                # Informações da palavra atual
                x, y, w, h = data['left'][j], data['top'][j], data['width'][j], data['height'][j]
                palavra_info = {
                    'texto': texto_palavra,
                    'bbox': [x, y, x + w, y + h]  # [x1, y1, x2, y2]
                }

                # Adiciona a palavra à sua respectiva linha
                if chave_linha not in linhas_agrupadas:
                    linhas_agrupadas[chave_linha] = []
                linhas_agrupadas[chave_linha].append(palavra_info)

        # Agora, processa as linhas agrupadas para criar os Documentos
        for chave_linha, palavras_na_linha in linhas_agrupadas.items():
            if not palavras_na_linha:
                continue

            # Junta o texto de todas as palavras da linha
            texto_da_linha = ' '.join([p['texto'] for p in palavras_na_linha])

            # Calcula a bounding box que engloba a linha inteira
            x1 = min([p['bbox'][0] for p in palavras_na_linha])
            y1 = min([p['bbox'][1] for p in palavras_na_linha])
            x2 = max([p['bbox'][2] for p in palavras_na_linha])
            y2 = max([p['bbox'][3] for p in palavras_na_linha])
            bbox_linha = [x1, y1, x2, y2]

            # Cria o objeto Document do LangChain
            doc = Document(
                page_content=texto_da_linha,
                metadata={
                    "source": f"Página {chave_linha[0]}",
                    "bbox": bbox_linha
                }
            )
            documentos_completos.append(doc)

    return documentos_completos


# ==============================================================================
# ETAPA 3: FUNÇÕES DE RAG (DIVISÃO, BANCO VETORIAL, BUSCA E LLM)
# ==============================================================================

def dividir_em_chunks(documentos_ocr: list):
    """Divide os documentos extraídos em chunks menores."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documentos_ocr)


def criar_faiss(chunks: list[Document]):
    """Cria um índice FAISS usando similaridade de cosseno."""
    embedder_model = carregar_modelo_embedding()
    texts = [doc.page_content for doc in chunks]

    print("--- GERANDO EMBEDDINGS COM BAAI/bge-m3 ---")
    embeddings = embedder_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("--- BANCO VETORIAL FAISS CRIADO COM SUCESSO ---")
    return index, chunks


def configurar_llm():
    """Configura e retorna a cadeia LangChain com o LLM."""
    
    model = "gemma3:4b"
    llm = ChatOllama(model=model, temperature=0)
    prompt_template = """
        Você é um assistente preciso para extrair informações de documentos.
        Com base no CONTEXTO fornecido, extraia os seguintes dados:
        
        - Nome da empresa: ...
        - Número do NIRE: ...
        - Nomes das pessoas no documento: ...
        - CPF de cada pessoa: ...
        - Endereço da sede da empresa: ...
        - Número de protocolo: ...
        - Data do documento: ...
        
        Responda APENAS com a informação presente no CONTEXTO. 
        Se uma informação não for encontrada, escreva 'Não especificado'.
        Não adicione nenhuma informação ou formatação que não foi solicitada.
        
        CONTEXTO:
        {context}
        
        TAREFA: {question}
        
        RESPOSTA:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    return (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def buscar_resposta(index, documentos, query_llm, chain):
    """
    Busca no índice FAISS e passa o contexto para o LLM gerar uma resposta.
    'query_llm' é a instrução completa para o LLM.
    """
    
    # --- INÍCIO DA CORREÇÃO ---
    # 1. Criamos uma "Query de Busca" otimizada para o FAISS.
    #    Ela foca nos TERMOS e CONCEITOS, não nas instruções.
    query_busca_faiss = "Nome da empresa, registro comercial, NIRE, nomes de pessoas, sócios, CPF, endereço da sede, número de protocolo, data do documento"
    
    print(f"\n--- DEBUG: Query Original (LLM): {query_llm}")
    print(f"--- DEBUG: Query Otimizada (FAISS): {query_busca_faiss}")
    
    # 2. Usamos a query_busca_faiss para gerar o embedding da busca
    embedder_model = carregar_modelo_embedding()
    query_embedding = embedder_model.encode(
        [query_busca_faiss], convert_to_numpy=True).astype("float32")
    # --- FIM DA CORREÇÃO ---

    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, 5) # Tentamos buscar 5 chunks

    retrieved_docs_obj = [documentos[i] for i in I[0] if i < len(documentos)]

    if not retrieved_docs_obj:
        return {"resposta": "Nenhuma informação relevante encontrada.", "fontes": [], "documentos_fonte": [], "bboxes": []}

    # --- DEBUG: Vamos ver os documentos que essa nova query encontrou ---
    print("\n--- DEBUG: Documentos Recuperados (Pós-Correção) ---")
    for i, doc in enumerate(retrieved_docs_obj):
        print(f"--- Doc {i} (Fonte: {doc.metadata.get('source')}) ---")
        print(f"Conteúdo: {doc.page_content[:150]}...")
    print("--------------------------------------------------")
    # --- FIM DEBUG ---

    fontes = {doc.metadata.get("source", "Desconhecida")
              for doc in retrieved_docs_obj}
    context_text = "\n\n".join([d.page_content for d in retrieved_docs_obj])

    print("\n--- DEBUG: NOVO CONTEXTO ENVIADO AO LLM ---")
    print(context_text)
    print("------------------------------------")

    # 3. Usamos a query_llm (a original, longa) para o LLM
    resposta_llm = chain.invoke({"context": context_text, "question": query_llm})

    bboxes_relevantes = []
    
    resposta_texto = resposta_llm.lower()
    palavras_da_resposta = set(re.findall(r'\w+', resposta_texto))
    palavras_ignoradas = {"não", "nao", "especificado", "documento", "paginas", "da", "de", "do", "para", "o", "a", "com", "em"}
    palavras_relevantes = palavras_da_resposta - palavras_ignoradas

    for doc in retrieved_docs_obj:
        chunk_texto = doc.page_content.lower()
        
        if any(palavra in chunk_texto for palavra in palavras_relevantes):
            try:
                page_num_str = re.search(r'\d+', doc.metadata["source"]).group(0)
                bboxes_relevantes.append({
                    "page": int(page_num_str) - 1, 
                    "bbox": doc.metadata["bbox"]
                })
            except Exception as e:
                print(f"AVISO: Falha ao extrair bbox/página do metadata: {doc.metadata}. Erro: {e}")

    # Converte 'documentos_fonte' (objetos) para dicts (serializável)
    documentos_fonte_serializaveis = []
    for doc in retrieved_docs_obj:
        documentos_fonte_serializaveis.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })

    resposta_llm_dict = {
        "resposta": resposta_llm,
        "fontes": sorted(list(fontes)),
        "documentos_fonte": documentos_fonte_serializaveis, 
        "bboxes": bboxes_relevantes
    }

    return resposta_llm_dict

# ==============================================================================
# ETAPA 4: ORQUESTRADOR PRINCIPAL (AGORA SÍNCRONO)
# ==============================================================================


def fluxo_principal(arquivo_bytes: bytes, nome_arquivo: str):
    """
    Orquestra todo o fluxo de forma síncrona, usando funções cacheadas.
    """
    # 1. Chama a função cacheada que faz todo o trabalho pesado
    # A chamada agora é direta, sem asyncio.
    print("==============================================")
    print("DEBUG: [fluxo_principal] INICIADO")
    print(f"DEBUG: Recebido arquivo: {nome_arquivo}, Tamanho: {len(arquivo_bytes)} bytes")
    
    try:
        # 1. Tentar processar o documento
        print("DEBUG: [Etapa 1] Chamando processar_documento_cacheado...")
        index, chunks_final = processar_documento_cacheado(
            arquivo_bytes, nome_arquivo
        )
        print("DEBUG: [Etapa 1] processar_documento_cacheado CONCLUÍDO.")

        # 2. Verificar o resultado do processamento
        if index is None:
            print("DEBUG: [Etapa 1] FALHA. 'index' é None. Documento pode ser inválido ou vazio.")
            return {
                "resposta": "Não foi possível extrair texto do documento. Verifique a qualidade da imagem ou do PDF.",
                "fontes": [],
                "documentos_fonte": []
            }
        
        print(f"DEBUG: [Etapa 1] SUCESSO. 'index' criado. Total de chunks: {len(chunks_final)}")

        # 3. Preparar a busca
        query = "Extraia as informações do documento solicitadas no template, com base no contexto fornecido."
        print(f"DEBUG: [Etapa 2] Configurando LLM e chain...")
        chain = configurar_llm()
        print("DEBUG: [Etapa 2] CONCLUÍDO.")

        # 4. Executar a busca
        print("DEBUG: [Etapa 3] Chamando buscar_resposta (LLM)...")
        resposta = buscar_resposta(index, chunks_final, query, chain)
        print("DEBUG: [Etapa 3] buscar_resposta CONCLUÍDO.")

        # 5. Logar a resposta final (antes de retornar)
        print("----------------------------------------------")
        print("DEBUG: [Resultado] RESPOSTA BRUTA DO BACKEND:")
        print(resposta) # O print que você já tinha
        print("----------------------------------------------")
        
        # 6. Verificação final
        if resposta is None:
            print("DEBUG: [Resultado] AVISO: 'buscar_resposta' retornou None.")
            # Retorna um dicionário de erro claro em vez de None
            return {
                "resposta": "Erro: A função de busca (LLM) não retornou nada (None).",
                "fontes": [],
                "documentos_fonte": []
            }

        print("DEBUG: [fluxo_principal] RETORNANDO RESPOSTA COM SUCESSO.")
        print("==============================================")
        return resposta

    except Exception as e:
        # Captura QUALQUER erro que aconteceu nas etapas acima
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"DEBUG: [fluxo_principal] !!! EXCEÇÃO CAPTURADA !!!")
        print(f"Erro: {e}")
        traceback.print_exc() # Imprime o stack trace completo no console
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Retorna um dicionário de erro claro para o frontend
        return {
            "resposta": f"Erro crítico no backend: {e}",
            "fontes": [],
            "documentos_fonte": []
        }



# ==============================================================================
#  ORQUESTRADOR PRINCIPAL 
# ==============================================================================
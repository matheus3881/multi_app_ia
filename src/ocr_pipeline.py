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
    # CORREÇÃO: Usar dpi=200 para UI e dpi=300 para OCR. 
    # O app.py usa 200, então o pipeline de OCR deve usar 300.
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for pagina in doc:
            pix = pagina.get_pixmap(dpi=300) # 300 DPI para melhor OCR
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
        data = pytesseract.image_to_data(img_cv, lang='por', output_type=Output.DICT) # lang='por'

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
                    "source": f"Página {chave_linha[0]}", # Chave: "Página N"
                    "page": chave_linha[0] - 1, # Chave: índice 0
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
    
    model = "gemma3:4b" # gemma3:4b não existe, gemma:2b ou gemma:7b
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
    
    query_busca_faiss = "Nome da empresa, registro comercial, NIRE, nomes de pessoas, sócios, CPF, endereço da sede, número de protocolo, data do documento"
    
    print(f"\n--- DEBUG: Query Original (LLM): {query_llm}")
    print(f"--- DEBUG: Query Otimizada (FAISS): {query_busca_faiss}")
    
    embedder_model = carregar_modelo_embedding()
    query_embedding = embedder_model.encode(
        [query_busca_faiss], convert_to_numpy=True).astype("float32")

    faiss.normalize_L2(query_embedding)

    k = min(5, len(documentos)) # Garante que k não seja maior que o nro de docs
    D, I = index.search(query_embedding, k) 

    retrieved_docs_obj = [documentos[i] for i in I[0] if i < len(documentos)]

    if not retrieved_docs_obj:
        return {"resposta": "Nenhuma informação relevante encontrada.", "fontes": [], "documentos_fonte": [], "bboxes": []}

    print("\n--- DEBUG: Documentos Recuperados (Pós-Correção) ---")
    for i, doc in enumerate(retrieved_docs_obj):
        print(f"--- Doc {i} (Fonte: {doc.metadata.get('source')}) ---")
        print(f"Conteúdo: {doc.page_content[:150]}...")
    print("--------------------------------------------------")

    fontes = {doc.metadata.get("source", "Desconhecida")
              for doc in retrieved_docs_obj}
    context_text = "\n\n".join([d.page_content for d in retrieved_docs_obj])

    print("\n--- DEBUG: NOVO CONTEXTO ENVIADO AO LLM ---")
    print(context_text)
    print("------------------------------------")

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
                # Fluxo OCR: Adiciona BBox se existir
                if "bbox" in doc.metadata and "page" in doc.metadata:
                    bboxes_relevantes.append({
                        "page": int(doc.metadata["page"]), # Usa o índice 0
                        "bbox": doc.metadata["bbox"]
                    })
            except Exception as e:
                print(f"AVISO: Falha ao extrair bbox/página do metadata: {doc.metadata}. Erro: {e}")

    # MODIFICADO: Retorna os objetos Document originais
    # Isso é necessário para a função de realce de PDF puro (Problema 2)
    # E não quebra a função de BBox
    resposta_llm_dict = {
        "resposta": resposta_llm,
        "fontes": sorted(list(fontes)),
        "documentos_fonte": retrieved_docs_obj, # Retorna os objetos
        "bboxes": bboxes_relevantes
    }

    return resposta_llm_dict

# ==============================================================================
# ETAPA 4: ORQUESTRADOR PRINCIPAL (OCR)
# ==============================================================================


def fluxo_principal(arquivo_bytes: bytes, nome_arquivo: str):
    """
    (FLUXO OCR) Orquestra todo o fluxo de forma síncrona.
    """
    print("==============================================")
    print("DEBUG: [fluxo_principal - OCR] INICIADO")
    print(f"DEBUG: Recebido arquivo: {nome_arquivo}, Tamanho: {len(arquivo_bytes)} bytes")
    
    try:
        print("DEBUG: [Etapa 1] Chamando processar_documento_cacheado...")
        index, chunks_final = processar_documento_cacheado(
            arquivo_bytes, nome_arquivo
        )
        print("DEBUG: [Etapa 1] processar_documento_cacheado CONCLUÍDO.")

        if index is None:
            print("DEBUG: [Etapa 1] FALHA. 'index' é None.")
            return {
                "resposta": "Não foi possível extrair texto do documento. Verifique a qualidade da imagem ou do PDF.",
                "fontes": [], "documentos_fonte": [], "bboxes": []
            }
        
        print(f"DEBUG: [Etapa 1] SUCESSO. 'index' criado. Total de chunks: {len(chunks_final)}")

        query = "Extraia as informações do documento solicitadas no template, com base no contexto fornecido."
        print(f"DEBUG: [Etapa 2] Configurando LLM e chain...")
        chain = configurar_llm()
        print("DEBUG: [Etapa 2] CONCLUÍDO.")

        print("DEBUG: [Etapa 3] Chamando buscar_resposta (LLM)...")
        resposta = buscar_resposta(index, chunks_final, query, chain)
        print("DEBUG: [Etapa 3] buscar_resposta CONCLUÍDO.")

        print("----------------------------------------------")
        print("DEBUG: [Resultado] RESPOSTA BRUTA DO BACKEND (OCR):")
        # print(resposta) # Log muito grande, talvez logar só a resposta
        print(resposta.get("resposta"))
        print("----------------------------------------------")
        
        if resposta is None:
            print("DEBUG: [Resultado] AVISO: 'buscar_resposta' retornou None.")
            return {
                "resposta": "Erro: A função de busca (LLM) não retornou nada (None).",
                "fontes": [], "documentos_fonte": [], "bboxes": []
            }

        print("DEBUG: [fluxo_principal - OCR] RETORNANDO RESPOSTA COM SUCESSO.")
        print("==============================================")
        return resposta

    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"DEBUG: [fluxo_principal - OCR] !!! EXCEÇÃO CAPTURADA !!!")
        print(f"Erro: {e}")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        return {
            "resposta": f"Erro crítico no backend: {e}",
            "fontes": [], "documentos_fonte": [], "bboxes": []
        }

# ==============================================================================
# INÍCIO: NOVAS FUNÇÕES ADICIONADAS (PARA PDF PURO)
# ==============================================================================

@st.cache_data
def processar_texto_puro_cacheado(pure_text_pages: list[str], file_name: str):
    """
    (NOVA E CORRIGIDA) "Mestra" do Texto Puro: Lista de Páginas -> Chunks -> Index
    Reutiliza as funções 'dividir_em_chunks' e 'criar_faiss'.
    """
    print(f"\n--- EXECUTANDO PROCESSAMENTO DE TEXTO (CACHE MISS) PARA: {file_name} ---")
    
    # 1. Empacotar o texto (página por página) em objetos Document
    # CORREÇÃO: Itera sobre a lista de páginas
    documentos_base = []
    for i, page_text in enumerate(pure_text_pages):
        doc = Document(
            page_content=page_text,
            metadata={
                "source": f"Página {i+1}", # Metadata CORRETO para os botões
                "page": i # Índice 0
            }
        )
        documentos_base.append(doc)
    
    if not documentos_base:
        print("--- AVISO: Nenhum texto recebido do PDF puro. ---")
        return None, None
    
    # 2. Dividir o texto em chunks menores (REUTILIZANDO SUA FUNÇÃO)
    chunks = dividir_em_chunks(documentos_base)

    if not chunks:
        print("--- AVISO: Nenhum chunk de texto foi gerado (Texto Puro). ---")
        return None, None

    # 3. Criar o índice vetorial FAISS (REUTILIZANDO SUA FUNÇÃO)
    index, chunks_final = criar_faiss(chunks)

    return index, chunks_final

def fluxo_principal_texto_puro(pure_text_pages: list[str], nome_arquivo: str):
    """
    (NOVO E CORRIGIDO) Orquestra o fluxo de RAG para PDFs baseados em texto.
    Reutiliza 'configurar_llm' e 'buscar_resposta'.
    """
    print("==============================================")
    print(f"DEBUG: [fluxo_principal_texto_puro] INICIADO para {len(pure_text_pages)} páginas")
    
    try:
        # 1. Chama a nova função cacheada de processamento de texto
        print("DEBUG: [Etapa 1] Chamando processar_texto_puro_cacheado...")
        index, chunks_final = processar_texto_puro_cacheado(
            pure_text_pages, nome_arquivo # Passa a lista de páginas
        )
        print("DEBUG: [Etapa 1] processar_texto_puro_cacheado CONCLUÍDO.")

        if index is None:
            print("DEBUG: [Etapa 1] FALHA. 'index' é None.")
            return {
                "resposta": "Não foi possível processar o texto do documento.",
                "fontes": [], "documentos_fonte": [], "bboxes": []
            }
        
        print(f"DEBUG: [Etapa 1] SUCESSO. 'index' criado. Total de chunks: {len(chunks_final)}")

        # 2. Preparar a busca (REUTILIZANDO SUA LÓGICA)
        query = "Extraia as informações do documento solicitadas no template, com base no contexto fornecido."
        print(f"DEBUG: [Etapa 2] Configurando LLM e chain...")
        chain = configurar_llm()
        print("DEBUG: [Etapa 2] CONCLUÍDO.")

        # 3. Executar a busca (REUTILIZANDO SUA FUNÇÃO)
        print("DEBUG: [Etapa 3] Chamando buscar_resposta (LLM)...")
        resposta = buscar_resposta(index, chunks_final, query, chain)
        print("DEBUG: [Etapa 3] buscar_resposta CONCLUÍDO.")
        
        # O fluxo de texto puro não gera bboxes de OCR
        resposta["bboxes"] = [] 

        print("----------------------------------------------")
        print("DEBUG: [Resultado] RESPOSTA BRUTA DO BACKEND (Texto Puro):")
        # print(resposta) # Log muito grande
        print(resposta.get("resposta"))
        print("----------------------------------------------")
        
        if resposta is None:
            return {
                "resposta": "Erro: A função de busca (LLM) não retornou nada (None).",
                "fontes": [], "documentos_fonte": [], "bboxes": []
            }

        print("DEBUG: [fluxo_principal_texto_puro] RETORNANDO RESPOSTA COM SUCESSO.")
        print("==============================================")
        return resposta

    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"DEBUG: [fluxo_principal_texto_puro] !!! EXCEÇÃO CAPTURADA !!!")
        print(f"Erro: {e}")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {
            "resposta": f"Erro crítico no backend (texto puro): {e}",
            "fontes": [], "documentos_fonte": [], "bboxes": []
        }

# ==============================================================================
# FIM: NOVAS FUNÇÕES ADICIONADAS
# ==============================================================================

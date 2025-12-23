import traceback
from turtle import onclick
import streamlit as st
import io
import re
from PIL import Image
import pymupdf  # Fitz
from st_clipboard import copy_to_clipboard_unsecured
from PIL import ImageDraw
import time

# --- Importações dos seus módulos ---
from src.ocr_pipeline import fluxo_principal, fluxo_principal_texto_puro
from src.sqlServer_search import buscar_no_banco
from src.transcricao import transcrever_audio


st.set_page_config(layout="wide")
st.title("Plataforma Multi-função com IA")

# ==============================================================================
# INICIALIZAÇÃO DO SESSION STATE
# ==============================================================================
st.session_state.setdefault("ocr_results", None)
st.session_state.setdefault("vector_results", None)
st.session_state.setdefault("pagina_atual_idx", 0)
st.session_state.setdefault("last_uploaded_filename", None)
st.session_state.setdefault("file_bytes", None)
st.session_state.setdefault("imagens_originais", None)
st.session_state.setdefault("imagens_processadas", None)
# --- MODIFICADO: Renomeado para 'pure_text_pages' ---
st.session_state.setdefault("is_pure_pdf", False)
st.session_state.setdefault("pure_text_pages", None)  # Agora é uma lista
st.session_state.setdefault("pdf_realcado_bytes", None)
st.session_state.setdefault("total_time", None)


# ==============================================================================
# INÍCIO: FUNÇÕES DE LÓGICA DO "BRAÇO" PDF PURO
# ==============================================================================


def detectar_pdf_puro(pdf_bytes: bytes) -> tuple[bool, list[str] | None]:
    """
    Detecta se um PDF é "puro" (baseado em texto).

    CORRIGIDO: Retorna (True, [texto_pag1, texto_pag2, ...]) se for puro.
    Retorna (False, None) se for escaneado.
    """
    textos_por_pagina = []
    is_puro = False
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if len(doc) == 0:
                return False, None  # PDF vazio

            # Checa as 5 primeiras páginas para decidir
            max_paginas_check = min(len(doc), 5)
            for i in range(max_paginas_check):
                page = doc.load_page(i)
                texto_pagina = page.get_text("text")
                texto_limpo = re.sub(r"\s+", "", texto_pagina).strip()

                # O limite de 100 caracteres é o que define se há uma camada de texto
                if len(texto_limpo) > 100:
                    is_puro = True
                    break

            # Se foi detectado como puro, extrai o texto de TODAS as páginas
            if is_puro:
                for page in doc:
                    textos_por_pagina.append(page.get_text("text"))
                return True, textos_por_pagina

    except Exception as e:
        st.error(f"Erro ao analisar o PDF: {e}")
        return False, None

    # Se não encontrou texto (ou não passou no 'if')
    return False, None


def adicionar_realce_pdf(pdf_bytes: bytes, trechos_para_realcar: list[str]) -> bytes:
    """
    CORRIGIDO: Pega os bytes do PDF original e adiciona realces com base
    em uma lista de strings (os *valores* extraídos pelo LLM).
    """
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        total_realces = 0

        if not trechos_para_realcar:
            st.warning("Nenhum valor extraído para realçar.")
            return pdf_bytes

        # Itera sobre cada valor (ex: "Empresa X", "12345")
        for trecho in trechos_para_realcar:
            # Itera sobre cada página do documento
            for page in doc:
                # Procura o texto exato
                coordenadas_encontradas = page.search_for(trecho.strip(), quads=True)

                if coordenadas_encontradas:
                    total_realces += len(coordenadas_encontradas)
                    for coord in coordenadas_encontradas:
                        page.add_highlight_annot(coord)

        if total_realces > 0:
            st.toast(f"{total_realces} realces adicionados ao PDF.")
            output_bytes = doc.tobytes()
            doc.close()
            return output_bytes
        else:
            st.toast(
                "Não foi possível encontrar os valores extraídos no PDF para realce."
            )
            doc.close()
            return pdf_bytes

    except Exception as e:
        st.error(f"Erro ao adicionar realce ao PDF: {e}")
        return pdf_bytes


# ==============================================================================
# FIM: NOVAS FUNÇÕES
# ==============================================================================

# ==============================================================================
# DEFINIÇÃO DOS SEPARADORES (TABS)
# ==============================================================================
tab_ocr, tab_vetorial, tab_transcricao = st.tabs(
    ["Análise de Documentos (OCR+RAG)", "Busca Vetorial", "Transcrição"]
)

with tab_ocr:
    st.header("Extraia e converse com seus documentos")
    uploaded_file = st.file_uploader(
        "Envie um PDF ou imagem", type=["pdf", "png", "jpg", "jpeg"], key="ocr_uploader"
    )

    # --- ETAPA 1: CARREGAR E CONVERTER O ARQUIVO ---
    if uploaded_file is not None:
        if st.session_state.last_uploaded_filename != uploaded_file.name:
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.pagina_atual_idx = 0

            file_bytes = uploaded_file.getvalue()
            st.session_state.file_bytes = file_bytes

            st.session_state.imagens_originais = None
            st.session_state.imagens_processadas = None
            st.session_state.ocr_results = None
            st.session_state.pdf_realcado_bytes = None
            st.session_state.is_pure_pdf = False
            st.session_state.pure_text_pages = None

            imagens_convertidas = []
            if uploaded_file.type == "application/pdf":
                # --- INÍCIO: LÓGICA DE DETECÇÃO (MODIFICADA) ---
                eh_puro, textos_por_pagina = detectar_pdf_puro(file_bytes)
                st.session_state.is_pure_pdf = eh_puro
                st.session_state.pure_text_pages = (
                    textos_por_pagina  # Agora é uma lista
                )
                # --- FIM: LÓGICA DE DETECÇÃO ---

                # Converte para imagens (para exibição inicial)
                with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
                    for page in doc:
                        pix = page.get_pixmap(dpi=200)  # dpi=200 é mais rápido para UI
                        imagens_convertidas.append(
                            Image.open(io.BytesIO(pix.tobytes()))
                        )
            else:
                # É imagem
                st.session_state.is_pure_pdf = False
                st.session_state.pure_text_pages = None
                imagens_convertidas.append(Image.open(io.BytesIO(file_bytes)))

            st.session_state.imagens_originais = imagens_convertidas

    # --- ETAPA 2: PROCESSAMENTO (AÇÃO DO BOTÃO) ---
    if st.button("Processar Documento"):
        if st.session_state.file_bytes:
            start_time = time.perf_counter()
            st.session_state.total_time = None

            with st.spinner("Analisando o documento..."):
                try:
                    # --- INÍCIO: LÓGICA DE "BRAÇOS" ---

                    if st.session_state.get("is_pure_pdf", False):
                        # --- BRAÇO 1: PDF PURO (COM CAMADA DE TEXTO) ---
                        st.info(
                            "Detectada camada de texto. Processando texto existente..."
                        )

                        resposta_rag = fluxo_principal_texto_puro(
                            st.session_state.pure_text_pages,  # Passa a lista de páginas
                            st.session_state.last_uploaded_filename,
                        )
                        st.session_state.ocr_results = resposta_rag

                        # --- LÓGICA DE REALCE (PDF PURO) - CORRIGIDA ---
                        if resposta_rag and resposta_rag.get("resposta"):

                            # 1. Parsear a resposta do LLM para extrair os *valores*
                            valores_para_realcar = []
                            linhas = resposta_rag["resposta"].strip().split("\n")
                            for linha in linhas:
                                if ":" in linha:
                                    try:
                                        partes = linha.split(":", 1)
                                        valor = partes[1].strip()
                                        # Não realça "Não especificado"
                                        if valor.lower() not in [
                                            "não especificado",
                                            "nao especificado",
                                            "",
                                        ]:
                                            valores_para_realcar.append(valor)
                                    except Exception:
                                        pass  # Ignora linhas mal formatadas

                            if valores_para_realcar:
                                st.write("Gerando PDF com realces...")
                                # 2. Passar os *valores* para a função de realce
                                pdf_realcado_bytes = adicionar_realce_pdf(
                                    st.session_state.file_bytes,
                                    valores_para_realcar,  # Passa a lista de valores
                                )
                                st.session_state.pdf_realcado_bytes = pdf_realcado_bytes

                                # 3. Converte o PDF realçado em imagens para exibição
                                st.toast("Atualizando visualização com realces...")
                                imagens_realcadas = []
                                with pymupdf.open(
                                    stream=pdf_realcado_bytes, filetype="pdf"
                                ) as doc_realcado:
                                    for page in doc_realcado:
                                        pix = page.get_pixmap(
                                            dpi=200
                                        )  # Manter DPI consistente
                                        imagens_realcadas.append(
                                            Image.open(io.BytesIO(pix.tobytes()))
                                        )

                                # Guarda as imagens JÁ REALÇADAS em 'imagens_processadas'
                                st.session_state.imagens_processadas = imagens_realcadas
                                st.session_state.pagina_atual_idx = 0
                            else:
                                st.toast(
                                    "Nenhum valor extraído da resposta para realçar."
                                )

                        else:
                            st.session_state.pdf_realcado_bytes = None
                            st.warning(
                                "Resposta do RAG não contém 'resposta'. Não é possível realçar."
                            )

                    else:
                        # --- BRAÇO 2: PDF ESCANEADO (SEM TEXTO) / IMAGEM ---
                        st.info("Camada de texto não detectada. Executando OCR...")

                        st.session_state.pdf_realcado_bytes = None
                        st.session_state.imagens_processadas = None

                        resposta_rag = fluxo_principal(
                            st.session_state.file_bytes,
                            st.session_state.last_uploaded_filename,
                        )
                        st.session_state.ocr_results = resposta_rag
                        # A lógica de desenhar 'bboxes' já está na ETAPA 3

                    # --- FIM: LÓGICA DE "BRAÇOS" ---

                except Exception as e:
                    st.session_state.ocr_results = None
                    st.error(f"Erro ao processar o documento: {e}")
                    print("--- ERRO NO FLUXO PRINCIPAL ---")
                    traceback.print_exc()
                    print("-------------------------------")
                finally:
                    end_time = time.perf_counter()
                    total_time = end_time - start_time
                    st.session_state.total_time = total_time
        else:
            st.warning("Por favor, envie um arquivo primeiro.")

        if st.session_state.get("total_time") is not None:
            st.info(
                f"Tempo total de processamento: {st.session_state.total_time:.2f} segundos"
            )
            st.session_state.total_time = None

    # --- ETAPA 3: EXIBIÇÃO EM DUAS COLUNAS ---
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Documento Analisado")

        # Esta linha agora suporta os dois fluxos:
        # 1. Fluxo Puro: usa 'imagens_processadas' (já realçadas por PyMuPDF)
        # 2. Fluxo OCR: usa 'imagens_originais' (e o ImageDraw desenha bboxes por cima)
        imagens_para_mostrar = st.session_state.get(
            "imagens_processadas"
        ) or st.session_state.get("imagens_originais")
        pagina_idx = st.session_state.pagina_atual_idx

        if imagens_para_mostrar:
            if len(imagens_para_mostrar) > 1:
                opcoes_de_pagina = [
                    f"Página {i+1}" for i in range(len(imagens_para_mostrar))
                ]

                if st.session_state.pagina_atual_idx >= len(opcoes_de_pagina):
                    st.session_state.pagina_atual_idx = 0

                pagina_idx = st.session_state.pagina_atual_idx  # Atualiza o índice

                def atualizar_pagina():
                    st.session_state.pagina_atual_idx = opcoes_de_pagina.index(
                        st.session_state.selectbox_pagina
                    )

                st.selectbox(
                    "Navegue pelas páginas:",
                    options=opcoes_de_pagina,
                    index=pagina_idx,
                    on_change=atualizar_pagina,
                    key="selectbox_pagina",
                )

            # --- LÓGICA DE DESTAQUE (BBOX) PARA FLUXO OCR ---
            # Para o fluxo de PDF Puro, 'bboxes_todas' será [] e nada será desenhado.

            ocr_results = st.session_state.get("ocr_results")
            bboxes_todas = []

            if ocr_results:
                bboxes_todas = ocr_results.get("bboxes", [])

            # 1. Pega a imagem (original ou já realçada)
            imagem_original = imagens_para_mostrar[pagina_idx]
            imagem_para_desenhar = imagem_original.convert("RGBA")
            draw = ImageDraw.Draw(imagem_para_desenhar)

            # 2. Filtra os bboxes (só vai encontrar algo no fluxo de OCR)
            lista_de_bboxes = [
                item["bbox"] for item in bboxes_todas if item["page"] == pagina_idx
            ]

            # 3. Itera e desenha (só no fluxo OCR)
            for bbox_lista in lista_de_bboxes:
                try:
                    x0, y0, x1, y1 = bbox_lista
                    rect_coords = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
                    draw.rectangle(rect_coords, outline="yellow", width=5)

                except Exception as e:
                    st.error(f"Erro ao desenhar bbox {bbox_lista}: {e}")

            # 4. Mostra a imagem final
            st.image(imagem_para_desenhar, width="content")

        else:
            st.info("Aguardando o envio de um documento...")

    with col2:
        st.subheader("Informações Extraídas")

        if st.session_state.get("ocr_results"):
            resultados = st.session_state.ocr_results
            resposta_str = resultados.get("resposta", "Nenhuma resposta gerada.")

            # --- LÓGICA DE EXIBIÇÃO (igual, sem mudança) ---
            linhas_da_resposta = resposta_str.strip().split("\n")

            for i, linha in enumerate(linhas_da_resposta):
                if ":" in linha:
                    partes = linha.split(":", 1)
                    topico = partes[0].replace("-", "").strip()
                    valor = partes[1].strip()

                    if valor.lower() not in ["não especificado", "nao especificado"]:
                        col_texto, col_botao = st.columns([4, 1])
                        with col_texto:
                            st.markdown(f"**{topico}**")
                            st.code(valor, language=None)
                        with col_botao:
                            st.markdown("&nbsp;")
                            if st.button("Copiar", key=f"copy_btn_{i}"):
                                copy_to_clipboard_unsecured(valor)
                                st.toast(f'"{valor}" copiado!', icon="📋")
                    else:
                        st.markdown(f"**{topico}**: *{valor}*")

                else:
                    st.write(linha)

            # --- LÓGICA DAS FONTES (CORRIGIDA) ---
            # Esta lógica agora funciona para AMBOS os fluxos,
            # pois 'fontes' sempre conterá 'Página N'
            st.divider()
            st.markdown("**Fontes Interativas:**")

            fontes_str = resultados.get("fontes", [])
            if fontes_str:
                numeros_paginas_encontrados = []
                for f in fontes_str:
                    # Procura por dígitos na string (ex: "Página 1")
                    match = re.search(r"\d+", str(f))
                    if match:
                        numeros_paginas_encontrados.append(int(match.group(0)))

                numeros_paginas = sorted(list(set(numeros_paginas_encontrados)))

                if numeros_paginas:
                    # Ajusta o número de colunas para não estourar
                    cols_botoes = st.columns(min(len(numeros_paginas), 5))

                    for i, num_pag in enumerate(numeros_paginas):
                        with cols_botoes[i % 5]:  # Usa módulo para distribuir

                            def mudar_pagina(idx, label):
                                st.session_state.pagina_atual_idx = idx
                                if "selectbox_pagina" in st.session_state:
                                    st.session_state.selectbox_pagina = label

                            label_pagina = f"Página {num_pag}"
                            idx_pagina = num_pag - 1  # Converte página 1 para índice 0

                            # Verifica se o índice da página é válido
                            if (
                                0
                                <= idx_pagina
                                < len(st.session_state.get("imagens_originais", []))
                            ):
                                st.button(
                                    f"Página {num_pag}",
                                    on_click=mudar_pagina,
                                    args=(idx_pagina, label_pagina),
                                    key=f"btn_pag_{num_pag}",
                                )
                            else:
                                st.caption(f"Pág. {num_pag} (Inválida)")
                else:
                    st.caption("Nenhuma fonte específica foi identificada.")
            else:
                st.caption("Nenhuma fonte específica foi identificada.")
        else:
            st.info("Aguardando o processamento para exibir os resultados.")


# ==============================================================================
with tab_vetorial:
    # (O seu código da tab_vetorial permanece intacto)
    st.header("Consulte a base de dados de conhecimento")
    query = st.text_input("O que você gostaria de saber?", key="vector_input")
    top_k = st.number_input("Número de resultados", min_value=1, max_value=10, value=5)

    if st.button("Buscar"):
        if query:
            with st.spinner("Buscando na base vetorial..."):
                resposta_busca = buscar_no_banco(query, top_k)
                st.session_state.vector_results = resposta_busca
        else:
            st.warning("Por favor, digite uma pergunta.")

    if st.session_state.vector_results:
        st.subheader("Resultados da Busca:")
        st.write(st.session_state.vector_results)

# ============================================================================
with tab_transcricao:
    # inicializar botão
    st.header("Transcreva seus arquivos de áudio")

    audio_file = st.file_uploader(
        "Escolha um arquivo de áudio",
        type=["mp3", "ogg",],
        key="transcricao_audio",
    )

    if audio_file is not None:
        # Verificação de tamanho (apenas visual, não bloqueia no seu código original)
        if audio_file.size > 26214400:
            st.warning(
                f"⚠️ O arquivo é muito grande! ({audio_file.size / (1024*1024):.2f} MB). Isso pode demorar."
            )

        st.success("Arquivo carregado e pronto.")

        # se audio existe botão deve existir

        if st.button("Transcrever", key="transcrever"):
            # se botão for clicado, botão deve ser desativa

            
            with st.spinner("Wait for it...", show_time=True):
                try:
                    # Chamada da função protegida
                    texto_final = transcrever_audio(audio_file)


                    st.divider()
                    st.subheader("Resultado:")
                    st.write(texto_final)

                except RuntimeError:
                    st.error("Ocorreu um erro interno. Verifique os logs do sistema.")

    else:
        st.info("👆 Por favor, carregue um arquivo de áudio para começar.")

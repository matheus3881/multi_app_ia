import traceback
import streamlit as st
import io
import re
from PIL import Image
import pymupdf  # Fitz
from st_clipboard import copy_to_clipboard_unsecured
from PIL import ImageDraw

# --- Importações dos seus módulos ---
# (Assumimos que src.ocr_pipeline e src.sqlServer_search existem)
from src.ocr_pipeline import fluxo_principal
from src.sqlServer_search import buscar_no_banco


st.set_page_config(layout="wide")
st.title("Plataforma Multi-função com IA")

# ==============================================================================
# INICIALIZAÇÃO DO SESSION STATE
# ==============================================================================
st.session_state.setdefault('ocr_results', None)
st.session_state.setdefault('vector_results', None)
st.session_state.setdefault('pagina_atual_idx', 0)
st.session_state.setdefault('last_uploaded_filename', None)
st.session_state.setdefault('file_bytes', None)
st.session_state.setdefault('imagens_originais', None)
st.session_state.setdefault('imagens_processadas', None)

# ==============================================================================
# DEFINIÇÃO DOS SEPARADORES (TABS)
# ==============================================================================
tab_ocr, tab_vetorial = st.tabs(["Análise de Documentos (OCR+RAG)", "Busca Vetorial"])

with tab_ocr:
    st.header("Extraia e converse com seus documentos")
    uploaded_file = st.file_uploader(
        "Envie um PDF ou imagem",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        key="ocr_uploader"
    )

    # --- ETAPA 1: CARREGAR E CONVERTER O ARQUIVO (CORREÇÃO DO ERRO) ---
    if uploaded_file is not None:
        if st.session_state.last_uploaded_filename != uploaded_file.name:
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.pagina_atual_idx = 0
            
            # Guarda os bytes no session_state para serem acessados pelo botão
            file_bytes = uploaded_file.getvalue()
            st.session_state.file_bytes = file_bytes
            
            # Converte o ficheiro (PDF ou imagem) numa lista de imagens PIL
            imagens_convertidas = []
            if uploaded_file.type == "application/pdf":
                with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
                    for page in doc:
                        pix = page.get_pixmap(dpi=200)
                        imagens_convertidas.append(Image.open(io.BytesIO(pix.tobytes())))
            else:
                imagens_convertidas.append(Image.open(io.BytesIO(file_bytes)))
            
            st.session_state.imagens_originais = imagens_convertidas
            st.session_state.imagens_processadas = None
            st.session_state.ocr_results = None

    # --- ETAPA 2: PROCESSAMENTO (AÇÃO DO BOTÃO) ---
    if st.button("Processar Documento"):
        if st.session_state.file_bytes:
            with st.spinner("Analisando o documento..."):
                # A chamada agora é síncrona e lê os dados do session_state
                try:
                    resposta_rag = fluxo_principal(
                        st.session_state.file_bytes,
                        st.session_state.last_uploaded_filename
                    )
                    st.session_state.ocr_results = resposta_rag
                    # Futura lógica de destaque virá aqui

                except Exception as e:
                     # Se 'fluxo_principal' falhar, limpa os resultados e MOSTRA O ERRO
                    st.session_state.ocr_results = None 
                    st.error(f"Erro ao processar o documento: {e}")
                    # Loga o erro completo no console para debug
                    print("--- ERRO NO FLUXO PRINCIPAL ---")
                    traceback.print_exc()
                    print("-------------------------------")
                # *** FIM DA CORREÇÃO ***
        else:
            st.warning("Por favor, envie um ficheiro primeiro.")
            
    # --- ETAPA 3: EXIBIÇÃO EM DUAS COLUNAS ---
    col1, col2 = st.columns(2, gap='large')

    with col1:
        st.subheader("Documento Analisado")
        imagens_para_mostrar = st.session_state.get('imagens_processadas') or st.session_state.get('imagens_originais')
        pagina_idx = st.session_state.pagina_atual_idx

        if imagens_para_mostrar:
            if len(imagens_para_mostrar) > 1:
                opcoes_de_pagina = [f"Página {i+1}" for i in range(len(imagens_para_mostrar))]
                
                if "pagina_atual_idx" not in st.session_state:
                    st.session_state.pagina_atual_idx = 0
                elif st.session_state.pagina_atual_idx >= len(opcoes_de_pagina):
                    st.session_state.pagina_atual_idx = 0

                def atualizar_pagina():
                    st.session_state.pagina_atual_idx = opcoes_de_pagina.index(st.session_state.selectbox_pagina)

                st.selectbox(
                    "Navegue pelas páginas:",
                    options=opcoes_de_pagina,
                    index=pagina_idx,
                    on_change=atualizar_pagina,
                    key='selectbox_pagina'
                )

            # --- CORREÇÃO: CONSTRUINDO OS BBOXES A PARTIR DE 'documentos_fonte' ---
            
            ocr_results = st.session_state.get("ocr_results")
            bboxes_todas = [] # Inicializa a lista

            if ocr_results:
                # O backend já preparou a lista 'bboxes'. Nós só precisamos lê-la.
                bboxes_todas = ocr_results.get("bboxes", [])
            
            
            # 1. Pega a imagem original e converte para RGBA
            imagem_original = imagens_para_mostrar[pagina_idx]
            imagem_para_desenhar = imagem_original.convert("RGBA")
            draw = ImageDraw.Draw(imagem_para_desenhar)

            # 2. Filtra os bboxes que acabamos de construir
            lista_de_bboxes = [item["bbox"] for item in bboxes_todas if item["page"] == pagina_idx]

            # 3. Itera e desenha
            for bbox_lista in lista_de_bboxes:
                try:
                    # O bbox já deve ser uma lista [x0, y0, x1, y1]
                    x0, y0, x1, y1 = bbox_lista
                    rect_coords = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
                    draw.rectangle(rect_coords, outline="yellow", width=5)
                    
                except Exception as e:
                    st.error(f"Erro ao desenhar bbox {bbox_lista}: {e}")

            # 4. Mostra a imagem final com os destaques
            st.image(imagem_para_desenhar, width="content")
            
        else:
            st.info("Aguardando o envio de um documento...")

    with col2:
        st.subheader("Informações Extraídas")
        
        if st.session_state.get('ocr_results'):
            resultados = st.session_state.ocr_results
            resposta_str = resultados.get("resposta", "Nenhuma resposta gerada.")

            # --- LÓGICA DE EXIBIÇÃO CORRIGIDA ---
            linhas_da_resposta = resposta_str.strip().split('\n')
            
            for i, linha in enumerate(linhas_da_resposta):
                if ':' in linha:
                    partes = linha.split(':', 1)
                    topico = partes[0].replace('-', '').strip()
                    valor = partes[1].strip()

                    # *** INÍCIO DA CORREÇÃO ***
                    if valor.lower() != 'não especificado':
                        # Se tiver um valor real, mostre com o botão de cópia
                        col_texto, col_botao = st.columns([4, 1])

                        with col_texto:
                            st.markdown(f"**{topico}**")
                            st.code(valor, language=None)

                        with col_botao:
                            st.markdown("&nbsp;") 
                    
                            if st.button("Copiar", key=f"copy_btn_{i}"):
                                copy_to_clipboard_unsecured(valor)
                                st.toast(f'"{valor}" copiado!', icon='📋')
                    else:
                        # Se o valor for "Não especificado", apenas exiba-o
                        # sem o botão de cópia.
                        st.markdown(f"**{topico}**: *{valor}*")
                    # *** FIM DA CORREÇÃO ***
                
                else:
                    # Exibe linhas que não são chave-valor (como linhas em branco ou cabeçalhos)
                    st.write(linha)

            # A exibição das fontes interativas continua igual
            st.divider()
            st.markdown("**Fontes Interativas:**")
            
            fontes_str = resultados.get("fontes", [])
            if fontes_str:
                # Extrai números das strings de fonte (ex: "Página 1")
                numeros_paginas_encontrados = []
                for f in fontes_str:
                    match = re.search(r'\d+', str(f))
                    if match:
                        numeros_paginas_encontrados.append(int(match.group(0)))

                numeros_paginas = sorted(list(set(numeros_paginas_encontrados)))
                
                if numeros_paginas:
                    cols_botoes = st.columns(len(numeros_paginas) or 1)
                    
                    for i, num_pag in enumerate(numeros_paginas):
                        with cols_botoes[i]:
                            def mudar_pagina(idx, label):
                                st.session_state.pagina_atual_idx = idx
                                if 'selectbox_pagina' in st.session_state:
                                    st.session_state.selectbox_pagina = label

                            label_pagina = f"Página {num_pag}"
                            # Verifica se o índice da página é válido
                            idx_pagina = num_pag - 1
                            if 0 <= idx_pagina < len(st.session_state.get('imagens_originais', [])):
                                st.button(
                                    f"Página {num_pag}", 
                                    on_click=mudar_pagina, 
                                    args=(idx_pagina, label_pagina),
                                    key=f"btn_pag_{num_pag}"
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
        # Supondo que a função de busca retorne algo que possa ser exibido diretamente
        st.write(st.session_state.vector_results)

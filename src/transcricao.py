import ssl
import os
import tempfile
import logging
import time
import streamlit as st

# --- BLOCO DE SEGURANÇA (Adicione ISTO na linha 1, antes de importar o whisper) ---
# Isso obriga o Python a aceitar certificados da sua rede corporativa
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Versões muito antigas do Python não tem isso (não é seu caso)
    pass
else:
    # Aqui é a mágica: Trocamos o contexto padrão seguro por um "sem verificação"
    ssl._create_default_https_context = _create_unverified_https_context

# --- 1. Configuração do Logger ---
# Isso define o formato: [HORA] - [NÍVEL] - MENSAGEM
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app_log.log"),  # Salva em arquivo
        logging.StreamHandler(),  # Mostra no terminal
    ],
)

# Criamos um logger específico para este módulo
logger = logging.getLogger(__name__)

import whisper

@st.cache_resource(show_spinner=False)
def carregar_modelo():
    model = whisper.load_model("turbo")
    return model


def transcrever_audio(audio_file: str):
    inicio = time.perf_counter()
    file_name = audio_file.name
    # file_size_mb = audio_file.size / (1024 * 1024)

    # logger.info(
    #     f"Iniciando transcrição: Arquivo='{file_name}' | Tamanho={file_size_mb:.2f}MB"
    # )

    # suffix = os.path.splitext(audio_file.name)[1]
    # fd, temp_path = tempfile.mkstemp(suffix=suffix)

    try:
        # with os.fdopen(fd, "wb") as tmp:
        #     tmp.write(audio_file.read())

        #     audio_file.seek(0)


        # Whisper recebe o PATH, não fd
        model_whinsper = carregar_modelo()
        result = model_whinsper.transcribe(audio_file)

        fim = time.perf_counter()

        duracao = fim - inicio

        print(f"O bloco de código levou {duracao:.4f} segundos para ser executado")

        return result["text"]

    except Exception as e:
        logger.error(
            f"Falha na transcrição: Arquivo='{file_name}' | Erro: {str(e)}",
            exc_info=True,
        )
        raise RuntimeError(f"Erro ao processar o áudio: {e}") from e

    # finally:
    #     if os.path.exists(temp_path):
    #         os.remove(temp_path)
    #         logger.debug(f"Arquivo temporário removido: {temp_path}")

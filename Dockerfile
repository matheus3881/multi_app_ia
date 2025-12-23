# Use a imagem base 'bullseye' (Debian 11) que é estável
FROM python:3.11-bullseye

# Define o diretório de trabalho dentro do container
WORKDIR /app

# --- Dependências do Sistema (Parte 1: Ferramentas Básicas) ---
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    libleptonica-dev \
    g++ \
    unixodbc-dev \
    curl \
    gnupg \
    ca-certificates \
    apt-transport-https \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- CORREÇÃO PROXY/SSL (Parte 1.5) ---
RUN echo 'Acquire::https::Verify-Peer "false";' > /etc/apt/apt.conf.d/99-insecure-ssl

# --- Dependências do Sistema (Parte 2: Chave Microsoft) ---
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSLk https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg \
    && chmod 644 /etc/apt/keyrings/microsoft.gpg

# --- Dependências do Sistema (Parte 3: Repositório Microsoft) ---
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/11/prod bullseye main" > /etc/apt/sources.list.d/mssql-release.list

# --- Dependências do Sistema (Parte 4: Instalação do Driver) ---
RUN apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia APENAS o arquivo de dependências primeiro.
COPY requirements.txt .

# Atualiza o pip
RUN pip install --no-cache-dir --upgrade pip

# --- OTIMIZAÇÃO DE TAMANHO + CORREÇÃO SSL/PIP ---
# Adiciona '--trusted-host download.pytorch.org' para o firewall
RUN pip install --no-cache-dir \
    --trusted-host download.pytorch.org \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# --- CORREÇÃO SSL/PIP ---
# Adiciona '--trusted-host' para os repositórios padrão do pip
# (pypi.org, etc.) que também serão bloqueados pelo firewall.
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Agora, copia o restante dos arquivos da aplicação.
COPY . .


# Comando para iniciar (com o baseUrlPath que definimos)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=5002", "--server.baseUrlPath=/multi-app"]

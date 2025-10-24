import cv2
import pytesseract
from pytesseract import Output

# Define o caminho para o executável do Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\msantos\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# 1. Este é o *caminho* para a imagem, não uma lista
caminho_imagem = "images/page_7.png"

# 2. Carrega a imagem usando o OpenCV
# É nesta variável 'img' que vamos desenhar
try:
    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em '{caminho_imagem}'.")
        print("Verifique se o caminho está correto.")
        exit()
except Exception as e:
    print(f"Ocorreu um erro ao ler a imagem: {e}")
    exit()

# 3. Roda o Tesseract na imagem *carregada* (img)
data = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(data["text"])

print(f"Encontradas {n_boxes} caixas. Desenhando retângulos...")

# 4. Loop para desenhar todas as caixas
for i in range(n_boxes):
    # Vamos desenhar apenas caixas que têm um texto real (confiança > 0)
    # data["conf"][i] == -1 geralmente é o bloco de texto inteiro
    if int(data["conf"][i]) > 0: 
        # Coordenadas
        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]

        # Cantos
        top_left = (x, y)
        bottom_right = (x + w, y + h)

        # Parâmetros da caixa
        green = (0, 255, 0)
        thickness = 2  # Aumentei para 2px para ficar mais visível

        # Desenha o retângulo DIRETAMENTE na imagem 'img'.
        # A função modifica a imagem "in-place", não precisa de 'doc = ...'
        cv2.rectangle(img, top_left, bottom_right, green, thickness)

# 5. Exibe a imagem FINAL (depois que o loop terminou)
print("Processamento concluído. Mostrando imagem...")
cv2.imshow("Imagem Processada", img)

# 6. Espera o usuário pressionar qualquer tecla para fechar a janela
cv2.waitKey(0)

# 7. Fecha todas as janelas do OpenCV
cv2.destroyAllWindows()
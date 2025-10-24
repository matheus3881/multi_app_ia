import io
from pathlib import Path
import cv2
import fitz
import pymupdf
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\msantos\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


# def image_to_text(input_path):
#    """
#    A function to read text from images.
#    """
#    img = cv2.imread(input_path)
#    text = pytesseract.image_to_string(img)
#    return text.strip()


def draw_bounding_boxes(input_folder, output_folder):
     # Converte os caminhos de string para objetos Path (mais fácil de trabalhar)
    in_path = Path(input_folder)
    out_path = Path(output_folder)

    # 1. Garante que o diretório de saída exista
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Encontra todos os arquivos de imagem comuns no diretório de entrada
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = [f for f in in_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"Aviso: Nenhuma imagem encontrada no diretório '{input_folder}'")
        return

    print(f"Processando {len(image_files)} imagens de '{input_folder}'...")

    # 3. Processa cada imagem no diretório
    for input_file_path in image_files:
        try:
            # Carrega a imagem
            img = cv2.imread(str(input_file_path))
            if img is None:
                print(f"  - Erro ao ler {input_file_path.name}, pulando.")
                continue

            # Extrai dados (usei 'lang=lang' para português)
            data = pytesseract.image_to_data(img, output_type=Output.DICT,)
            n_boxes = len(data["text"])

            for i in range(n_boxes):
                # Desenha caixas apenas para texto com confiança razoável (ignora ruído)
                if int(data["conf"][i]) == -1:
                    # Coordenadas
                    x, y = data["left"][i], data["top"][i]
                    w, h = data["width"][i], data["height"][i]
                    # Cantos
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    # Parâmetros da caixa
                    green = (0, 255, 0)
                    thickness = 2 # Aumentei para 2 para ficar mais visível

                    cv2.rectangle(img, top_left, bottom_right, green, thickness)

            # 4. Define o caminho de saída (mesmo nome do arquivo original)
            output_file_path = out_path / input_file_path.name
            
            # 5. Salva a imagem com as caixas no diretório de saída
            cv2.imwrite(str(output_file_path), img)

        except Exception as e:
            print(f"  - Erro inesperado ao processar {input_file_path.name}: {e}")
    
    print(f"\nProcessamento concluído. Imagens salvas em '{output_folder}'.")


#  ================================= Converter PDF em Imagem ===========================================

# def convert_pdf_to_images(pdf_path, output_folder="output_images"):
#     """
#     Converts each page of a PDF document into a separate image file.

#     Args:
#         pdf_path (str): The path to the input PDF file.
#         output_folder (str): The directory where the image files will be saved.
#     """
#     try:
#         doc = fitz.open(pdf_path)
#     except fitz.FileNotFoundError:
#         print(f"Error: PDF file not found at '{pdf_path}'")
#         return

#     import os
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)  # Load a specific page
#         pix = page.get_pixmap()       # Render page to image

#         output_filename = os.path.join(output_folder, f"page_{page_num + 1}.png")
#         pix.save(output_filename)     # Save the image

#     doc.close()
#     print(f"Successfully converted '{pdf_path}' to images in '{output_folder}'")


# convert_pdf_to_images("contrato_finuse.pdf")



medium_text_path = "output_images"
output_path = "images"

draw_bounding_boxes(medium_text_path, output_path)


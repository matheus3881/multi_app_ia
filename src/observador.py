import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from transcricao import transcrever_audio

AUDIO_EXTS = {".ogg", ".mp3", ".wav", ".m4a"}


class MyHandler(FileSystemEventHandler):

    def on_created(self, event):

        path = event.src_path
        ext = os.path.splitext(path)[1].lower()

        # só processa arquivos de áudio
        if ext not in AUDIO_EXTS:
            print(f"Ignorando (não é áudio): {path}")
            return

        print(f"Created: {path}")
        texto = transcrever_audio(path)

        # nome_base = Path(event.src_path).stem

        # print(f"Extensão? {nome_base}")

        # grava o txt
        txt_path = f"{path}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(texto)

        print("Transcrição salva em:", txt_path)

    def on_modified(self, event):
        if not event.is_directory:
            print(f"Modified: {event.src_path}")

    def on_deleted(self, event):
        print(f"Deleted: {event.src_path}")


path = "./assets/"
event_handler = MyHandler()
observer = Observer()
observer.schedule(MyHandler, path, recursive=True)
observer.start()
try:
    while observer.is_alive():
        observer.join(1)
finally:
    observer.stop()
    observer.join()

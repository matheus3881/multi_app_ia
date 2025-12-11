import whisper


def whinsper(audio: bytes):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Path to your audio file
    audio_path = audio

    # Perform transcription
    result = model.transcribe(audio_path)


    # Print the transcribed text
    print("Transcription:")
    print(result["text"])

    return result["text"]





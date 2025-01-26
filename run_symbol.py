import time
from src.audio.recorder import AudioRecorder
from src.transcription.senseVoiceSmall import SenseVoiceSmallProcessor
# from src.llm.symbol import test
# from src.llm.translate import test
# from src.chat.deepseek import test
# from src.transcription.senseVoiceSmall import test
# from src.audio.recorder import test



if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.start_recording()
    time.sleep(5)
    audio_buffer = recorder.stop_recording()
    print(audio_buffer)

    senseVoiceSmall = SenseVoiceSmallProcessor()
    result = senseVoiceSmall.process_audio(audio_buffer)
    print(result)
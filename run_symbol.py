import time
from src.audio.recorder import AudioRecorder
from src.transcription.senseVoiceSmall import SenseVoiceSmallProcessor
from src.chat.deepseek import DeepSeekChat
from src.audio.text_to_speech import KokoroTTS
import sounddevice as sd
import numpy as np
from pynput import keyboard
from src.chat.stream_chat import StreamChat
# from src.audio.text_to_speech import KokoroTTS
# from src.llm.symbol import test

# from src.llm.translate import test
# from src.chat.deepseek import test
# from src.transcription.senseVoiceSmall import test
# from src.audio.recorder import test

class VoiceAssistant:
    def __init__(self):
        self.recorder = AudioRecorder()
        self.senseVoiceSmall = SenseVoiceSmallProcessor()
        self.deepseek = DeepSeekChat()
        self.tts = KokoroTTS()
        self.is_recording = False
        self.stream_chat = StreamChat()
        
    def on_press(self, key):
        """按键按下时的回调"""
        try:
            if key == keyboard.Key.alt and not self.is_recording:
                print("\n开始录音...")
                self.is_recording = True
                self.start_time = time.time()
                self.recorder.start_recording()
        except AttributeError:
            pass

    def on_release(self, key):
        """按键释放时的回调"""
        try:
            if key == keyboard.Key.alt and self.is_recording:
                print("停止录音...")
                self.is_recording = False
                
                # 1. 录音阶段
                record_start = time.time()
                audio_buffer = self.recorder.stop_recording()
                record_time = time.time() - record_start
                print(f"录音耗时: {record_time:.2f}秒")
                
                if audio_buffer == "TOO_SHORT":
                    print("录音时间太短")
                    return
                
                if not audio_buffer:
                    print("录音失败")
                    return

                # 2. 语音识别阶段
                asr_start = time.time()
                result, error = self.senseVoiceSmall.process_audio(audio_buffer)
                asr_time = time.time() - asr_start
                print(f"语音识别耗时: {asr_time:.2f}秒")
                
                if error:
                    print(f"语音识别失败: {error}")
                    return
                    
                print(f"语音识别结果: {result}")

                # 3. AI对话阶段（流式）
                chat_start = time.time()
                print("AI回复: ", end='', flush=True)
                
                current_text = ""
                for response in self.stream_chat.stream_chat(result):
                    print(response, end='', flush=True)
                    current_text += response
                    
                    # 4. 语音合成阶段（流式）
                    if any(char in response for char in '.!?。！？'):
                        tts_start = time.time()
                        try:
                            audio, _ = self.tts.speak(response)
                            tts_time = time.time() - tts_start
                            print(f"\n语音合成耗时: {tts_time:.2f}秒")
                        except Exception as e:
                            print(f"\n语音播放失败: {str(e)}")
                
                chat_time = time.time() - chat_start
                print(f"\nAI对话总耗时: {chat_time:.2f}秒")
                
                # 总耗时统计
                total_time = time.time() - self.start_time
                print(f"\n总耗时统计:")
                print(f"录音阶段: {record_time:.2f}秒 ({(record_time/total_time*100):.1f}%)")
                print(f"语音识别: {asr_time:.2f}秒 ({(asr_time/total_time*100):.1f}%)")
                print(f"AI对话: {chat_time:.2f}秒 ({(chat_time/total_time*100):.1f}%)")
                print(f"总耗时: {total_time:.2f}秒")
                    
        except AttributeError:
            pass

    def run(self):
        """运行语音助手"""
        print("语音助手已启动，按住 Option/Alt 键开始录音，松开键结束录音...")
        
        # 监听键盘事件
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release) as listener:
            listener.join()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()

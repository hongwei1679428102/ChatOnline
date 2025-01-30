import torch
import sounddevice as sd
import numpy as np
import os
from typing import Tuple, Optional
import requests
from tqdm import tqdm
from pathlib import Path
import sys
import soundfile as sf
from transformers import BertConfig, BertModel, BertTokenizer
import io

class KokoroTTS:
    MODEL_FILES = {
        'models.py': 'https://huggingface.co/hexgrad/Kokoro-82M/raw/main/models.py',
        'kokoro.py': 'https://huggingface.co/hexgrad/Kokoro-82M/raw/main/kokoro.py',
        'istftnet.py': 'https://huggingface.co/hexgrad/Kokoro-82M/raw/main/istftnet.py',
        'plbert.py': 'https://huggingface.co/hexgrad/Kokoro-82M/raw/main/plbert.py',
        'config.json': 'https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json',
        'kokoro-v0_19.pth': 'https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.pth',
        'voices/af.pt': 'https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af.pt'
    }
    
    def __init__(self):
        self.model_dir = Path(__file__).parent / 'Kokoro-82M'
        self._download_model_files()
        self._init_model()
    
    def _download_model_files(self):
        """下载必要的模型文件"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / 'voices').mkdir(exist_ok=True)
        (self.model_dir / 'bert').mkdir(exist_ok=True)
        
        # 下载 BERT 模型
        print("下载 BERT 模型...")
        bert_path = self.model_dir / 'bert'
        if not bert_path.exists():
            config = BertConfig.from_pretrained('bert-base-chinese')
            model = BertModel.from_pretrained('bert-base-chinese')
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            
            # 保存模型文件
            bert_path.mkdir(parents=True, exist_ok=True)
            config.save_pretrained(str(bert_path))
            model.save_pretrained(str(bert_path))
            tokenizer.save_pretrained(str(bert_path))
            print("BERT 模型下载完成")
        
        # 下载其他文件
        for filename, url in self.MODEL_FILES.items():
            file_path = self.model_dir / filename
            if not file_path.exists():
                print(f"下载模型文件: {filename}")
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    # 获取文件大小
                    total_size = int(response.headers.get('content-length', 0))
                    
                    # 确保父目录存在
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 使用进度条下载
                    with open(file_path, 'wb') as f, tqdm(
                        desc=filename,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        for data in response.iter_content(chunk_size=1024):
                            size = f.write(data)
                            pbar.update(size)
                    print(f"已下载: {filename}")
                except Exception as e:
                    raise Exception(f"下载失败 {filename}: {str(e)}")
    
    def _init_model(self):
        """初始化模型"""
        sys.path.append(str(self.model_dir))
        try:
            from models import build_model
            from kokoro import generate, generate_full, phonemize
            self.generate = generate  # 使用基础版本，更稳定
            self.phonemize = phonemize
        except ImportError as e:
            raise ImportError(f"无法导入必要的模块，请确保模型文件已正确下载。错误: {str(e)}")
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 构建模型
        model_file = self.model_dir / 'kokoro-v0_19.pth'
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
            
        self.model = build_model(str(model_file), self.device)
        print("模型加载完成")
        
        # 加载默认声音
        self.voice_name = 'af'  # Bella & Sarah 混合声音
        voice_file = self.model_dir / 'voices' / f'{self.voice_name}.pt'
        if not voice_file.exists():
            raise FileNotFoundError(f"声音文件不存在: {voice_file}")
            
        self.voicepack = torch.load(voice_file, weights_only=True).to(self.device)
        print(f"已加载声音: {self.voice_name}")

    def speak(self, text: str) -> Tuple[bytes, str]:
        """
        将文字转换为语音
        Args:
            text: 要转换的文字
        Returns:
            Tuple[bytes, str]: (音频数据, 音素)
        """
        try:
            # 预处理文本
            if not text or len(text.strip()) == 0:
                print("文本为空")
                return None, None
                
            # 清理文本，移除多余的空白和换行
            text = ' '.join(text.strip().split())
            
            # 将长文本分段处理
            max_length = 100  # 每段最大字符数
            segments = []
            
            # 按句子分割
            sentences = text.split('. ')
            current_segment = ''
            
            for sentence in sentences:
                if len(current_segment) + len(sentence) <= max_length:
                    current_segment += sentence + '. '
                else:
                    if current_segment:
                        segments.append(current_segment.strip())
                    current_segment = sentence + '. '
            
            if current_segment:
                segments.append(current_segment.strip())
            
            # 处理每个分段
            full_audio = []
            for segment in segments:
                print(f"正在生成语音: {segment}")
                
                # 生成音频
                audio, phonemes = self.generate(
                    self.model, 
                    segment, 
                    self.voicepack, 
                    lang='a',  # 使用通用语言代码
                    speed=1.0
                )
                
                if audio is not None and isinstance(audio, np.ndarray):
                    # 确保音频是单声道
                    if len(audio.shape) > 1:
                        audio = audio[:, 0]  # 只保留第一个通道
                    else:
                        audio = audio.reshape(-1)
                    
                    # 添加到完整音频
                    full_audio.append(audio)
                
                # 添加短暂停顿
                silence = np.zeros(int(24000 * 0.3))  # 0.3秒静音
                full_audio.append(silence)
            
            if full_audio:
                # 合并所有音频片段
                audio = np.concatenate(full_audio)
                
                # 确保音频数据是 float32 类型且在 [-1, 1] 范围内
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                if np.abs(audio).max() > 1:
                    audio = audio / np.abs(audio).max()
                
                # 转换为 int16
                audio_int16 = (audio * 32767).astype(np.int16)
                
                # 创建临时缓冲区保存为 WAV 格式
                buffer = io.BytesIO()
                sf.write(buffer, audio_int16, 24000, format='WAV', subtype='PCM_16')
                buffer.seek(0)
                
                # 读取 WAV 文件数据
                audio_bytes = buffer.read()
                buffer.close()
                
                print("语音生成完成")
                return audio_bytes, phonemes
            else:
                print("音频生成失败")
                return None, None
                
        except Exception as e:
            print(f"TTS 生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
            
    def __del__(self):
        """析构函数，确保清理资源"""
        sd.stop()  # 停止任何正在播放的音频

    def change_voice(self, voice_name: str):
        """
        更换说话人声音
        可用的声音:
        - af: Bella & Sarah 混合
        - af_bella: Bella
        - af_sarah: Sarah
        - am_adam: Adam
        - am_michael: Michael
        - bf_emma: Emma (英式)
        - bf_isabella: Isabella (英式)
        - bm_george: George (英式)
        - bm_lewis: Lewis (英式)
        - af_nicole: Nicole
        - af_sky: Sky
        """
        try:
            self.voice_name = voice_name
            voice_file = self.model_dir / 'voices' / f'{voice_name}.pt'
            if not voice_file.exists():
                raise FileNotFoundError(f"声音文件不存在: {voice_file}")
            
            self.voicepack = torch.load(voice_file, weights_only=True).to(self.device)
            print(f"已切换到声音: {voice_name}")
        except Exception as e:
            print(f"切换声音失败: {str(e)}")

def test():
    tts = KokoroTTS()
    
    # # 测试中文
    # print("\n测试中文:")
    # audio, phonemes = tts.speak("你好，我是语音助手。")
    # if audio is not None:
    #     with open("test_cn.wav", "wb") as f:
    #         f.write(audio)
    #     print(f"已生成中文测试音频: test_cn.wav")
    #     print(f"音素: {phonemes}")
    
    # 测试英文
    print("\n测试英文:")
    audio, phonemes = tts.speak("""
    Absolutely! Here's an interesting fact: The word "tarantula" comes from the Italian city of Taranto, where it was believed that a bite from a tarantula spider could induce a sort of madness. Dancers were known to dance frantically to ward off the effects of the bite, a practice called tarantism. The condition was indeed believed to be treatable through physical exertion, leading to the association of the spider with the term "tarantula."
    """)
    if audio is not None:
        with open("test_en.wav", "wb") as f:
            f.write(audio)
        print(f"已生成英文测试音频: test_en.wav")
        print(f"音素: {phonemes}")

if __name__ == "__main__":
    test()
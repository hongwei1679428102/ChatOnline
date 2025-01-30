import os
import json
import time
import httpx
from typing import AsyncGenerator
import logging
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(env_path)

logger = logging.getLogger(__name__)

class ErnieBot:
    def __init__(self):
        print("正在初始化 ErnieBot...")
        print(f"环境变量文件路径: {env_path}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"BAIDU_API_KEY: {os.getenv('BAIDU_API_KEY')}")
        print(f"BAIDU_SECRET_KEY: {os.getenv('BAIDU_SECRET_KEY')}")
        
        self.api_key = os.getenv("BAIDU_API_KEY")
        self.secret_key = os.getenv("BAIDU_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError(f"环境变量未正确加载。API Key: {self.api_key}, Secret Key: {self.secret_key}")
            
        self.access_token = None
        self.token_expires = 0
        self.conversation_history = []
        self._stop_streaming = False
        
    def stop_streaming(self):
        """停止当前的流式输出"""
        self._stop_streaming = True
        
    def reset(self):
        """重置所有状态"""
        self.conversation_history = []
        self._stop_streaming = False
        
    async def _get_access_token(self):
        """获取百度 API 访问令牌"""
        now = time.time()
        if self.access_token and now < self.token_expires:
            return self.access_token
            
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, params=params)
            response.raise_for_status()
            result = response.json()
            
        self.access_token = result["access_token"]
        self.token_expires = now + result["expires_in"] - 60  # 提前60秒刷新
        return self.access_token
        
    async def stream_chat(self, user_input: str) -> AsyncGenerator[str, None]:
        """流式对话"""
        try:
            if self._stop_streaming:
                return
                
            # 记录用户输入
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # 获取访问令牌
            access_token = await self._get_access_token()
            
            # 准备请求数据
            url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token={access_token}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": self.conversation_history,
                "stream": True,
                "temperature": 0.7,
                "top_p": 0.8
            }
            
            async with httpx.AsyncClient() as client:
                async with client.stream('POST', url, json=data, headers=headers) as response:
                    response.raise_for_status()
                    full_response = ""
                    current_sentence = ""  # 用于缓存当前句子
                    
                    async for line in response.aiter_lines():
                        if self._stop_streaming:
                            break
                            
                        if line.startswith("data: "):
                            try:
                                json_data = json.loads(line[6:])  # 去掉 "data: " 前缀
                                if not json_data.get("is_end", False):
                                    content = json_data.get("result", "")
                                    if content:
                                        current_sentence += content
                                        # 检查是否有完整的句子
                                        sentences = self._split_into_sentences(current_sentence)
                                        if sentences:
                                            # 输出完整的句子
                                            for sentence in sentences[:-1]:  # 除了最后一个不完整的句子
                                                if sentence.strip():
                                                    full_response += sentence
                                                    yield sentence
                                            # 保留最后一个可能不完整的句子
                                            current_sentence = sentences[-1]
                            except json.JSONDecodeError as e:
                                logger.error(f"解析响应出错: {e}")
                                continue
                    
                    # 输出最后一个句子（如果有的话）
                    if current_sentence and not self._stop_streaming:
                        full_response += current_sentence
                        yield current_sentence
                        
            # 如果没有被中断，记录完整的对话历史
            if not self._stop_streaming and full_response:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
        except Exception as e:
            logger.error(f"对话出错: {e}")
            yield f"Error: {str(e)}"

    def _split_into_sentences(self, text: str) -> list[str]:
        """将文本分割成句子"""
        # 定义句子结束标记
        sentence_endings = '.。!！?？'
        result = []
        current = ""
        
        for char in text:
            current += char
            if char in sentence_endings:
                # 找到一个完整的句子
                result.append(current)
                current = ""
        
        # 添加剩余的不完整句子（如果有的话）
        if current:
            result.append(current)
            
        return result
        
    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []

async def test():
    """测试函数"""
    bot = ErnieBot()
    
    print("测试开始...")
    user_input = "你好，请用简短的话介绍一下你自己。"
    print(f"\n用户: {user_input}")
    print("助手: ", end='', flush=True)
    
    async for response in bot.stream_chat(user_input):
        print(response, end='', flush=True)
    print("\n")
    
    # 测试对话历史
    user_input = "刚才说的很好，请继续。"
    print(f"用户: {user_input}")
    print("助手: ", end='', flush=True)
    
    async for response in bot.stream_chat(user_input):
        print(response, end='', flush=True)
    print("\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test()) 
import os
import httpx
import json
from typing import AsyncGenerator, Optional
import logging

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(console_handler)

class StreamChat:
    def __init__(self):
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("未设置 SILICONFLOW_API_KEY 环境变量")
        self.model = os.getenv("SILICONFLOW_TRANSLATE_MODEL", "THUDM/glm-4-9b-chat")
        self.conversation_history = []
        self._stop_streaming = False  # 添加停止标志
        
    def stop_streaming(self):
        """停止当前的流式输出"""
        self._stop_streaming = True
        
    def reset(self):
        """重置所有状态"""
        self.conversation_history = []
        self._stop_streaming = False
        
    async def stream_chat(self, user_input: str) -> AsyncGenerator[str, None]:
        """流式处理用户输入并返回回应"""
        try:
            if self._stop_streaming:
                return
                
            # 记录用户输入
            self.conversation_history.append({"role": "user", "content": user_input})
            full_response = ""  # 添加这一行
            
            # 准备请求数据
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": self.conversation_history,
                "stream": True
            }
            
            async with httpx.AsyncClient() as client:
                async with client.stream('POST', 'https://api.deepseek.com/v1/chat/completions', json=data, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if self._stop_streaming:  # 检查是否需要停止
                            return
                            
                        if line.strip():
                            try:
                                json_line = json.loads(line.removeprefix('data: '))
                                content = json_line['choices'][0]['delta'].get('content', '')
                                if content:
                                    full_response += content  # 累积响应
                                    yield content
                            except Exception as e:
                                print(f"Error parsing streaming response: {e}")
                                
            # 如果没有被中断，记录完整的对话历史
            if not self._stop_streaming:
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": full_response  # 改用 full_response 而不是 response_text
                })
                
        except Exception as e:
            print(f"Error in stream chat: {e}")
            yield f"Error: {str(e)}"
    
    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []

def test():
    import asyncio
    
    async def run_test():
        stream_chat = StreamChat()
        
        # 测试流式对话
        user_input = "tell me something about honey."
        print(f"\n用户: {user_input}\n")
        print("助手: ", end='', flush=True)
        
        async for response in stream_chat.stream_chat(user_input):
            print(response, end='', flush=True)
        print("\n")

    asyncio.run(run_test())

if __name__ == "__main__":
    test() 
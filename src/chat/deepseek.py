import os
import httpx
from ..utils.logger import logger

class DeepSeekChat:
    def __init__(self):
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("未设置 SILICONFLOW_API_KEY 环境变量")
        self.model = os.getenv("SILICONFLOW_TRANSLATE_MODEL", "THUDM/glm-4-9b-chat")
        self.conversation_history = []
        
    def chat(self, user_input: str) -> str:
        """处理用户输入并返回回应"""
        try:
            # 记录用户输入
            logger.info(f"用户: {user_input}")
            
            # 添加用户输入到对话历史
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # 准备请求数据
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": self.conversation_history,
                "temperature": 0.7,
                "max_tokens": 2000,
                "stream": False  # 关闭流式输出
            }
            
            # 调用 API
            response = httpx.post(
                "https://api.siliconflow.cn/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API 调用失败: {response.text}")
            
            # 获取回应文本
            assistant_message = response.json()["choices"][0]["message"]["content"]
            
            # 清理回应文本，移除重复内容
            cleaned_message = self._clean_response(assistant_message)
            
            # 记录助手回应
            logger.info(f"助手: {cleaned_message}")
            
            # 添加助手回应到对话历史
            self.conversation_history.append({"role": "assistant", "content": cleaned_message})
            
            return cleaned_message
            
        except Exception as e:
            error_msg = f"对话处理失败: {str(e)}"
            logger.error(error_msg)
            return f"抱歉，{error_msg}"
            
    def _clean_response(self, text: str) -> str:
        """清理回应文本，移除重复内容"""
        # 按行分割
        lines = text.strip().split('\n')
        
        # 如果只有一行，直接返回
        if len(lines) <= 1:
            return text.strip()
            
        # 找到最长的完整回复
        max_line = ''
        for line in lines:
            line = line.strip()
            if line and len(line) > len(max_line):
                max_line = line
                
        return max_line
    
    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = [] 

def test():
    deepseek = DeepSeekChat()
    print(deepseek.chat("你好，我是小明，今天天气很好，你吃了吗？"))

if __name__ == "__main__":
    test()
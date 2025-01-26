import os
import sys
from pathlib import Path
import httpx
from ..utils.logger import logger
from dotenv import load_dotenv

load_dotenv()

class SymbolProcessor:
    def __init__(self):
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("未设置 SILICONFLOW_API_KEY 环境变量")
        self.model = os.getenv("SILICONFLOW_ADD_SYMBOL_MODEL", "THUDM/glm-4-9b-chat")

    def add_symbol(self, text):
        """为输入的文本添加合适的标点符号"""
        logger.info("正在添加标点符号...")
        
        try:
            # 准备请求数据
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": """
                    Please add appropriate punctuation to the user's input and return it. 
                    Apart from this, do not add or modify anything else. 
                    Do not translate the user's input. 
                    Do not add any explanation. 
                    Do not answer the user's question and so on. 
                    Just output the user's input with punctuation!
                    """},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
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
            result = response.json()["choices"][0]["message"]["content"]
            return result

        except Exception as e:
            logger.error(f"添加标点符号失败: {str(e)}")
            return text, e

    def optimize_result(self, text):
        """优化识别结果"""
        # system_prompt = """
        # You are a content input optimizer.

        # Since the user's input is the result of speech recognition, there may be some obvious inaccuracies or errors.
        # Please optimize the user's input based on your knowledge.
        # If the user's speech recognition result is fine, no changes are necessary—just output it directly.
        # Additionally, the user's speech recognition input might lack necessary punctuation.
        # Please add the appropriate punctuation and return the final result.

        # Notice:
        #     •	We only need to optimize the user's input content; there is no need to answer the user's question!!!
        #     •	Do not add any explanation.
        #     •	Do not add any other content.
        #     •	Do not translate the user's input.
        # """

        system_prompt = """
        You are a speech recognition content input optimizer.
        Please optimize the user's input based on your knowledge.
        And add appropriate punctuation to the user's input.
        Do not change the user's language.
        Do not add any explanation.
        Do not add answer to the user's question,just output the optimized content.
        """
        try:
            logger.info(f"正在优化识别结果...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        )
            return response.choices[0].message.content
        except Exception as e:
            return text, e
        

def test():
    print("test")
    symbol_processor = SymbolProcessor()
    text = "你好，我是小明，今天天气很好，你吃了吗"
    result = symbol_processor.add_symbol(text)
    print(result)

def main():
    # 你的主要代码
    pass

if __name__ == "__main__":
    test()
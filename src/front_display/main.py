from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import json
import asyncio
import io
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(env_path)

# 获取当前文件所在目录
BASE_DIR = Path(__file__).resolve().parent

# 修改导入路径
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# 修改导入语句
from src.audio.recorder import AudioRecorder
from src.transcription.senseVoiceSmall import SenseVoiceSmallProcessor
from src.chat.stream_chat import StreamChat
from src.audio.text_to_speech import KokoroTTS
from src.chat.ernie_bot import ErnieBot

# 确保目录存在
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
static_dir.mkdir(parents=True, exist_ok=True)
templates_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 设置模板
templates = Jinja2Templates(directory=str(templates_dir))

# 存储活动的 WebSocket 连接
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def get(request: Request):
    """返回主页"""
    try:
        # 确保模板目录存在
        if not templates_dir.exists():
            templates_dir.mkdir(parents=True, exist_ok=True)
            
        # 确保 index.html 存在
        index_path = templates_dir / "index.html"
        if not index_path.exists():
            # 如果不存在，运行设置脚本创建文件
            from .setup import setup_front_display
            setup_front_display()
            
        # 使用模板渲染
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "url_for": app.url_path_for  # 传递 url_for 函数给模板
            }
        )
    except Exception as e:
        print(f"Error serving index.html: {e}")
        import traceback
        traceback.print_exc()
        return HTMLResponse(
            content=f"Error loading template: {str(e)}",
            status_code=500
        )

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """提供静态文件服务"""
    return FileResponse(static_dir / file_path)

@app.get("/debug")
async def debug():
    """返回调试信息"""
    try:
        static_files = list(static_dir.rglob("*"))
        template_files = list(templates_dir.rglob("*"))
        
        return {
            "base_dir": str(BASE_DIR),
            "static_dir": str(static_dir),
            "templates_dir": str(templates_dir),
            "static_exists": static_dir.exists(),
            "templates_exists": templates_dir.exists(),
            "static_files": [str(f.relative_to(static_dir)) for f in static_files if f.is_file()],
            "template_files": [str(f.relative_to(templates_dir)) for f in template_files if f.is_file()]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """处理 WebSocket 连接"""
    try:
        await manager.connect(websocket)
        print("WebSocket connected")
        
        recorder = AudioRecorder()
        sense_voice = SenseVoiceSmallProcessor()
        stream_chat = ErnieBot()
        tts = KokoroTTS()
        current_task = None
        is_connected = True
        
        while is_connected:
            try:
                message = await websocket.receive()
                message_type = message.get("type", "")
                
                if message_type == "websocket.disconnect":
                    print("Received disconnect message")
                    is_connected = False
                    break
                
                # 如果是停止命令
                if message_type == "websocket.receive" and message.get("text"):
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "stop":
                            print("Received stop command")
                            stream_chat.stop_streaming()
                            if current_task and not current_task.done():
                                current_task.cancel()
                                try:
                                    await current_task
                                except asyncio.CancelledError:
                                    pass
                            stream_chat.reset()
                            continue
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue
                
                # 如果是音频数据
                if message_type == "websocket.receive" and message.get("bytes"):
                    # 取消之前的任务
                    if current_task and not current_task.done():
                        current_task.cancel()
                        try:
                            await current_task
                        except asyncio.CancelledError:
                            pass
                    
                    received_audio_data = message["bytes"]  # 重命名变量
                    print("Received audio data, length:", len(received_audio_data))
                    
                    # 创建新的对话任务，传入音频数据
                    async def process_audio_task(audio_data):  # 添加参数
                        try:
                            # 将字节数据转换为 BytesIO 对象
                            audio_buffer = io.BytesIO(audio_data)
                            
                            # 处理音频
                            result, error = sense_voice.process_audio(audio_buffer)
                            if error:
                                print(f"Audio processing error: {error}")
                                if is_connected:
                                    await websocket.send_json({
                                        "type": "error",
                                        "message": str(error)
                                    })
                                return
                            
                            if result:
                                print(f"Transcription result: {result}")
                                if is_connected:
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "message": result
                                    })
                                
                                # 流式处理 AI 回复
                                current_response = ""
                                async for response in stream_chat.stream_chat(result):
                                    if not is_connected:
                                        break
                                        
                                    if response:
                                        print(f"Chat response chunk: {response}")
                                        current_response += response
                                        await websocket.send_json({
                                            "type": "chat",
                                            "message": response
                                        })
                                        
                                        # 如果是完整的句子，就进行语音合成
                                        if any(char in response for char in '.!?。！？'):
                                            print(f"Synthesizing speech for: {current_response}")
                                            try:
                                                tts_audio = tts.speak(current_response)  # 重命名变量
                                                if tts_audio and is_connected:
                                                    print("Sending synthesized audio")
                                                    await websocket.send_bytes(tts_audio[0])
                                            except Exception as e:
                                                print(f"TTS error: {e}")
                                            current_response = ""
                            else:
                                print("No transcription result")
                                
                        except Exception as e:
                            print(f"Error processing audio: {e}")
                            import traceback
                            traceback.print_exc()
                            if is_connected:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": str(e)
                                })
                    
                    # 创建任务时传入音频数据
                    current_task = asyncio.create_task(process_audio_task(received_audio_data))
                    await current_task
                    
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                is_connected = False
                break
            except Exception as e:
                print(f"Error in websocket loop: {e}")
                import traceback
                traceback.print_exc()
                if str(e).startswith("Cannot call \"receive\" once a disconnect"):
                    is_connected = False
                    break
                
    finally:
        if current_task and not current_task.done():
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass
        
        stream_chat.stop_streaming()
        manager.disconnect(websocket)
        try:
            await websocket.close()
        except:
            pass
        print("WebSocket connection cleaned up")

if __name__ == "__main__":
    # 打印调试信息
    print(f"Base directory: {BASE_DIR}")
    print(f"Static directory: {static_dir}")
    print(f"Templates directory: {templates_dir}")
    
    # 启动服务器，使用 websockets 后端
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        ws='websockets'
    ) 
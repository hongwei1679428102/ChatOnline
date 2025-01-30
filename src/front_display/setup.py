import os
from pathlib import Path

def setup_front_display():
    # 获取当前目录
    base_dir = Path(__file__).resolve().parent
    
    # 创建必要的目录
    dirs = [
        base_dir / "static" / "css",
        base_dir / "static" / "js",
        base_dir / "templates"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # 创建 style.css
    css_content = """body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f0f2f5;
    color: #333;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

.chat-box {
    background: white;
    border-radius: 15px;
    padding: 20px;
    height: 70vh;
    overflow-y: auto;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.message {
    margin-bottom: 15px;
    padding: 12px 16px;
    border-radius: 15px;
    max-width: 80%;
    word-wrap: break-word;
}

.user-message {
    background-color: #007AFF;
    color: white;
    margin-left: auto;
    margin-right: 10px;
}

.assistant-message {
    background-color: #f1f1f1;
    margin-right: auto;
    margin-left: 10px;
}

.controls {
    text-align: center;
    padding: 20px 0;
}

.record-button {
    background-color: #007AFF;
    color: white;
    border: none;
    padding: 15px 40px;
    border-radius: 25px;
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.record-button:hover {
    background-color: #0056b3;
    transform: translateY(-1px);
}

.record-button:active {
    background-color: #ff3b30;
    transform: translateY(1px);
}

.record-button.recording {
    background-color: #ff3b30;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}"""
    
    # 创建 main.js
    js_content = """let ws;
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// 初始化 WebSocket 连接
function initWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        switch(data.type) {
            case 'transcription':
                addMessage(data.message, 'user');
                break;
            case 'chat':
                addMessage(data.message, 'assistant');
                break;
            case 'error':
                console.error(data.message);
                break;
        }
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = function() {
        setTimeout(initWebSocket, 1000);
    };
}

// 添加消息到对话框
function addMessage(message, type) {
    const chatBox = document.getElementById('chatBox');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    messageDiv.textContent = message;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// 初始化录音功能
async function initRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = function(event) {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async function() {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            ws.send(await audioBlob.arrayBuffer());
            audioChunks = [];
        };
        
    } catch (error) {
        console.error('录音初始化失败:', error);
    }
}

// 设置录音按钮事件
document.getElementById('recordButton').addEventListener('mousedown', function() {
    if (!isRecording) {
        isRecording = true;
        this.classList.add('recording');
        this.textContent = '松开结束';
        audioChunks = [];
        mediaRecorder.start();
    }
});

document.getElementById('recordButton').addEventListener('mouseup', function() {
    if (isRecording) {
        isRecording = false;
        this.classList.remove('recording');
        this.textContent = '按住说话';
        mediaRecorder.stop();
    }
});

// 初始化
initWebSocket();
initRecording();"""
    
    # 创建 index.html
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>语音对话助手</title>
    <link rel="stylesheet" href="/static/css/style.css">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div class="container">
        <div class="chat-box" id="chatBox">
            <div class="message assistant-message">
                您好，我是您的AI助手，请按住按钮开始对话。
            </div>
        </div>
        
        <div class="controls">
            <button id="recordButton" class="record-button">
                按住说话
            </button>
        </div>
    </div>
    
    <script src="/static/js/main.js"></script>
</body>
</html>"""
    
    # 写入文件
    files = {
        base_dir / "static" / "css" / "style.css": css_content,
        base_dir / "static" / "js" / "main.js": js_content,
        base_dir / "templates" / "index.html": html_content
    }
    
    for file_path, content in files.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    setup_front_display() 
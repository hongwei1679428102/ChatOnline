let ws;
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let audioQueue = [];
let isPlaying = false;
let currentAudioContext = null;  // 保存当前的 AudioContext
let currentSource = null;        // 保存当前的音频源
let isStopped = false;          // 标记是否强制停止
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

// 停止所有音频播放
function stopAllAudio() {
    // 清空音频队列
    audioQueue = [];
    
    // 停止当前正在播放的音频
    if (currentSource) {
        try {
            currentSource.stop();
        } catch (e) {
            console.log("Error stopping current source:", e);
        }
    }
    
    // 关闭当前的 AudioContext
    if (currentAudioContext) {
        try {
            currentAudioContext.close();
        } catch (e) {
            console.log("Error closing AudioContext:", e);
        }
    }
    
    currentSource = null;
    currentAudioContext = null;
    isPlaying = false;
}

// 初始化 WebSocket 连接
function initWebSocket() {
    console.log("Initializing WebSocket connection...");
    
    if (ws && ws.readyState !== WebSocket.CLOSED) {
        console.log("WebSocket already exists");
        return;
    }
    
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = function() {
        console.log("WebSocket connection established");
        reconnectAttempts = 0;  // 重置重连次数
    };
    
    ws.onmessage = async function(event) {
        console.log("Received message:", event.data);
        if (isStopped) {
            console.log("Message ignored - stopped state");
            return;
        }

        try {
            // 如果是二进制数据（音频）
            if (event.data instanceof Blob) {
                console.log("Received audio data");
                audioQueue.push(event.data);
                if (!isPlaying) {
                    playNextAudio();
                }
                return;
            }

            // 如果是 JSON 数据
            const data = JSON.parse(event.data);
            console.log("Parsed message data:", data);
            switch(data.type) {
                case 'transcription':
                    addMessage(data.message, 'user');
                    break;
                case 'chat':
                    addMessage(data.message, 'assistant');
                    break;
                case 'error':
                    console.error('Server error:', data.message);
                    addMessage(`Error: ${data.message}`, 'error');
                    break;
            }
        } catch (e) {
            console.error('Error processing message:', e);
        }
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = function(event) {
        console.log("WebSocket connection closed", event);
        
        // 如果不是正常关闭，尝试重连
        if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts})...`);
            setTimeout(initWebSocket, 1000 * Math.min(reconnectAttempts, 5));
        } else if (reconnectAttempts >= maxReconnectAttempts) {
            console.log("Max reconnection attempts reached");
            addMessage("连接服务器失败，请刷新页面重试", 'error');
        }
    };
}

// 播放下一个音频
async function playNextAudio() {
    if (audioQueue.length === 0 || isPlaying || isStopped) {
        return;
    }

    isPlaying = true;
    const audioData = audioQueue.shift();

    try {
        currentAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await audioData.arrayBuffer();
        const audioBuffer = await currentAudioContext.decodeAudioData(arrayBuffer);
        
        currentSource = currentAudioContext.createBufferSource();
        currentSource.buffer = audioBuffer;
        currentSource.connect(currentAudioContext.destination);
        
        // 监听播放结束事件
        currentSource.onended = () => {
            isPlaying = false;
            currentSource = null;
            currentAudioContext.close();
            currentAudioContext = null;
            if (!isStopped) {
                playNextAudio();  // 只有在未停止的情况下才继续播放
            }
        };
        
        currentSource.start(0);
    } catch (e) {
        console.error('Error playing audio:', e);
        isPlaying = false;
        currentSource = null;
        if (currentAudioContext) {
            currentAudioContext.close();
        }
        currentAudioContext = null;
        if (!isStopped) {
            playNextAudio();
        }
    }
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
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                console.error('WebSocket not connected');
                return;
            }
            
            try {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                ws.send(await audioBlob.arrayBuffer());
                console.log("Audio data sent");
            } catch (e) {
                console.error('Error sending audio:', e);
            }
            audioChunks = [];
        };
        
    } catch (error) {
        console.error('录音初始化失败:', error);
    }
}

// 修改录音按钮事件处理
document.getElementById('recordButton').addEventListener('mousedown', async function() {
    console.log("Record button pressed");
    
    // 检查 WebSocket 连接
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        console.log("WebSocket not connected, attempting to reconnect");
        addMessage("正在连接服务器，请稍后再试", 'error');
        initWebSocket();
        return;
    }
    
    if (!isRecording && mediaRecorder) {
        // 如果正在进行对话，先停止当前对话
        if (isPlaying || audioQueue.length > 0) {
            console.log("Stopping current conversation");
            isStopped = true;
            stopAllAudio();
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'stop' }));
            }
            addMessage("对话已停止", 'system');
        }
        
        // 重置状态并开始新的录音
        console.log("Starting new recording");
        isStopped = false;
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
        console.log("Recording stopped");
    }
});

// 初始化
initWebSocket();
initRecording();
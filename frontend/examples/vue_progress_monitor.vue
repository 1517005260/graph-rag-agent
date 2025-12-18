<template>
  <div class="build-progress-monitor">
    <h1>🚀 图谱构建进度监控</h1>
    <p class="subtitle">WebSocket 实时推送 (Vue 3 Demo)</p>

    <!-- 连接状态 -->
    <div :class="['connection-status', connected ? 'connected' : 'disconnected']">
      {{ connected ? '已连接' : '未连接' }}
    </div>

    <!-- 控制按钮 -->
    <div class="controls">
      <button class="btn-primary" @click="connectWebSocket">连接 WebSocket</button>
      <button class="btn-success" @click="startBuild">触发构建</button>
      <button class="btn-danger" @click="stopBuild">停止构建</button>
    </div>

    <!-- 统计卡片 -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">{{ stats.l0Files }}</div>
        <div class="stat-label">L0 处理文件数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ stats.l1Tasks }}</div>
        <div class="stat-label">L1 提交任务数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ stats.entities }}</div>
        <div class="stat-label">实体数量</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ stats.relations }}</div>
        <div class="stat-label">关系数量</div>
      </div>
    </div>

    <!-- 进度条 -->
    <div class="progress-section">
      <div class="progress-item">
        <div class="progress-label">
          <span>{{ l0Progress.details || 'L0 快速索引' }}</span>
          <span>{{ l0Progress.percent }}%</span>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar" :style="{ width: l0Progress.percent + '%' }">
            {{ l0Progress.percent }}%
          </div>
        </div>
      </div>

      <div class="progress-item">
        <div class="progress-label">
          <span>{{ l1Progress.details || 'L1 深度索引' }}</span>
          <span>{{ l1Progress.percent }}%</span>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar" :style="{ width: l1Progress.percent + '%' }">
            {{ l1Progress.percent }}%
          </div>
        </div>
      </div>
    </div>

    <!-- 日志区域 -->
    <div class="logs-section">
      <div class="logs-title">📋 实时日志</div>
      <div class="logs-container">
        <div
          v-for="log in logs"
          :key="log.id"
          :class="['log-entry', `log-${log.level.toLowerCase()}`]"
        >
          [{{ log.timestamp }}] {{ log.message }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onUnmounted } from 'vue';

// Props
const props = defineProps({
  apiUrl: {
    type: String,
    default: 'http://localhost:8000'
  }
});

// 状态
const connected = ref(false);
const ws = ref(null);

const l0Progress = ref({ percent: 0, details: '' });
const l1Progress = ref({ percent: 0, details: '' });

const logs = ref([]);

const stats = ref({
  l0Files: 0,
  l1Tasks: 0,
  entities: 0,
  relations: 0
});

// 连接 WebSocket
const connectWebSocket = () => {
  const wsUrl = props.apiUrl.replace('http', 'ws') + '/build/ws/progress';

  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    addLog('WebSocket 已连接', 'INFO');
    return;
  }

  ws.value = new WebSocket(wsUrl);

  ws.value.onopen = () => {
    connected.value = true;
    addLog('✅ WebSocket 连接成功', 'INFO');
  };

  ws.value.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleMessage(data);
  };

  ws.value.onerror = (error) => {
    console.error('WebSocket 错误:', error);
    addLog('❌ WebSocket 连接错误', 'ERROR');
  };

  ws.value.onclose = () => {
    connected.value = false;
    addLog('WebSocket 连接已关闭', 'WARNING');
  };
};

// 处理 WebSocket 消息
const handleMessage = (data) => {
  console.log('收到消息:', data);

  switch (data.type) {
    case 'connected':
      addLog(data.message, 'INFO');
      break;

    case 'log':
      addLog(data.content, data.level || 'INFO');
      break;

    case 'progress':
      if (data.stage === 'l0_ingestion') {
        l0Progress.value = { percent: data.percent, details: data.details || '' };
      } else if (data.stage === 'l1_indexing') {
        l1Progress.value = { percent: data.percent, details: data.details || '' };
      }
      break;

    case 'status':
      addLog(`状态更新: ${data.status} - ${data.message}`, 'INFO');
      break;

    case 'file_status':
      addLog(`文件 ${data.file_path}: ${data.status} (${data.stage})`, 'INFO');
      break;

    case 'stats':
      stats.value = {
        l0Files: data.data.l0_files || 0,
        l1Tasks: data.data.l1_tasks || 0,
        entities: data.data.entities || 0,
        relations: data.data.relations || 0
      };
      break;

    case 'error':
      addLog(`错误: ${data.error}`, 'ERROR');
      break;
  }
};

// 添加日志
const addLog = (message, level = 'INFO') => {
  const timestamp = new Date().toLocaleTimeString();
  const logEntry = {
    id: Date.now() + Math.random(), // 确保唯一性
    timestamp,
    message,
    level
  };

  logs.value = [logEntry, ...logs.value.slice(0, 49)]; // 保留最新 50 条
};

// 触发构建
const startBuild = async () => {
  try {
    const response = await fetch(`${props.apiUrl}/build/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: 'incremental',
        force: false,
        skip_l0: false,
        skip_l1: false
      })
    });

    const result = await response.json();
    addLog(`构建任务: ${result.message}`, result.status === 'started' ? 'INFO' : 'WARNING');
  } catch (error) {
    addLog(`触发构建失败: ${error.message}`, 'ERROR');
  }
};

// 停止构建
const stopBuild = async () => {
  try {
    const response = await fetch(`${props.apiUrl}/build/stop`, { method: 'POST' });
    const result = await response.json();
    addLog(`停止构建: ${result.message}`, 'WARNING');
  } catch (error) {
    addLog(`停止构建失败: ${error.message}`, 'ERROR');
  }
};

// 组件卸载时关闭 WebSocket
onUnmounted(() => {
  if (ws.value) {
    ws.value.close();
  }
});
</script>

<style scoped>
.build-progress-monitor {
  max-width: 900px;
  margin: 0 auto;
  background: white;
  border-radius: 16px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  padding: 30px;
}

h1 {
  color: #333;
  margin-bottom: 10px;
  font-size: 28px;
}

.subtitle {
  color: #666;
  margin-bottom: 30px;
  font-size: 14px;
}

.connection-status {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 20px;
}

.connection-status.connected {
  background: #10b981;
  color: white;
}

.connection-status.disconnected {
  background: #ef4444;
  color: white;
}

.controls {
  display: flex;
  gap: 10px;
  margin-bottom: 30px;
}

button {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  color: white;
}

button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

button:active {
  transform: translateY(0);
}

.btn-primary {
  background: #3b82f6;
}

.btn-success {
  background: #10b981;
}

.btn-danger {
  background: #ef4444;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-bottom: 30px;
}

.stat-card {
  background: #f3f4f6;
  padding: 15px;
  border-radius: 8px;
  text-align: center;
}

.stat-value {
  font-size: 24px;
  font-weight: 700;
  color: #3b82f6;
  margin-bottom: 5px;
}

.stat-label {
  font-size: 12px;
  color: #6b7280;
}

.progress-section {
  margin-bottom: 30px;
}

.progress-item {
  margin-bottom: 20px;
}

.progress-label {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
  color: #333;
}

.progress-bar-container {
  width: 100%;
  height: 24px;
  background: #e5e7eb;
  border-radius: 12px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  transition: width 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 12px;
  font-weight: 600;
}

.logs-section {
  background: #f9fafb;
  border-radius: 8px;
  padding: 20px;
}

.logs-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #333;
}

.logs-container {
  max-height: 400px;
  overflow-y: auto;
}

.log-entry {
  padding: 8px 12px;
  margin-bottom: 6px;
  border-radius: 6px;
  font-size: 13px;
  font-family: 'Courier New', monospace;
}

.log-info {
  background: #dbeafe;
  color: #1e40af;
}

.log-warning {
  background: #fef3c7;
  color: #92400e;
}

.log-error {
  background: #fee2e2;
  color: #991b1b;
}
</style>

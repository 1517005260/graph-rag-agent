# 图谱构建进度监控 - 前端示例

本目录包含两种实时通信方案（WebSocket 和 SSE）的多种前端实现示例。

## 📁 文件说明

| 文件 | 技术栈 | 通信方式 | 说明 |
|------|--------|----------|------|
| `websocket_progress_monitor.html` | 原生 HTML/JS | WebSocket | 双向通信，功能最全 |
| `sse_progress_monitor.html` | 原生 HTML/JS | SSE | 单向推送，更简单轻量 |
| `react_progress_monitor.jsx` | React | WebSocket | React 组件示例 |
| `vue_progress_monitor.vue` | Vue 3 | WebSocket | Vue 3 组件示例 |

### 🆚 WebSocket vs SSE

| 特性 | WebSocket | SSE |
|------|-----------|-----|
| 通信方向 | 双向 | 单向（服务器→客户端） |
| 协议 | ws:// / wss:// | HTTP |
| 自动重连 | 需要手动实现 | 浏览器内置 |
| 浏览器支持 | 所有现代浏览器 | 所有现代浏览器 |
| 适用场景 | 需要双向交互 | 只需接收推送 |
| 实现复杂度 | 稍复杂 | 简单 |

**推荐使用场景**：
- 只需监控进度 → **使用 SSE**（更简单）
- 需要控制构建（暂停/恢复）→ 使用 WebSocket

## 🚀 快速开始

### 方式 1: 使用 SSE 版本（**推荐，最简单**）

1. **启动后端服务**：
   ```bash
   cd server/
   python main.py
   ```

2. **打开 SSE HTML 文件**：
   ```bash
   # 在浏览器中打开
   open frontend/examples/sse_progress_monitor.html
   # 或直接双击文件
   ```

3. **使用步骤**：
   - 点击「连接 SSE」按钮
   - 点击「触发构建」按钮开始构建
   - 实时查看进度条和日志更新
   - SSE 会自动重连，无需手动处理

**优点**：基于 HTTP，浏览器自动重连，代码更简单

### 方式 2: 使用 WebSocket 版本

1. **启动后端服务**：
   ```bash
   cd server/
   python main.py
   ```

2. **打开 WebSocket HTML 文件**：
   ```bash
   open frontend/examples/websocket_progress_monitor.html
   ```

3. **使用步骤**：
   - 点击「连接 WebSocket」按钮
   - 点击「触发构建」按钮开始构建
   - 实时查看进度条和日志更新

**优点**：双向通信，可扩展控制功能（暂停/恢复）

### 方式 2: 使用 React 版本

1. **复制组件到你的 React 项目**：
   ```bash
   cp frontend/examples/react_progress_monitor.jsx your-project/src/components/
   ```

2. **使用组件**：
   ```jsx
   import BuildProgressMonitor from './components/react_progress_monitor';

   function App() {
     return (
       <div className="App">
         <BuildProgressMonitor apiUrl="http://localhost:8000" />
       </div>
     );
   }
   ```

3. **确保项目支持 ES6+ 和 React Hooks**

### 方式 3: 使用 Vue 3 版本

1. **复制组件到你的 Vue 项目**：
   ```bash
   cp frontend/examples/vue_progress_monitor.vue your-project/src/components/
   ```

2. **使用组件**：
   ```vue
   <template>
     <div id="app">
       <BuildProgressMonitor :api-url="apiUrl" />
     </div>
   </template>

   <script setup>
   import BuildProgressMonitor from './components/vue_progress_monitor.vue';

   const apiUrl = 'http://localhost:8000';
   </script>
   ```

3. **确保项目使用 Vue 3 + Composition API**

## 📡 WebSocket 消息协议

前端通过 WebSocket 连接 `ws://localhost:8000/build/ws/progress` 接收以下类型的消息：

### 1. 连接成功
```json
{
  "type": "connected",
  "message": "WebSocket 连接成功",
  "timestamp": "2025-12-13T10:00:00"
}
```

### 2. 日志消息
```json
{
  "type": "log",
  "level": "INFO",  // INFO | WARNING | ERROR
  "content": "开始 L0 快速摄取流程",
  "timestamp": "2025-12-13T10:00:01"
}
```

### 3. 进度更新
```json
{
  "type": "progress",
  "stage": "l0_ingestion",  // l0_ingestion | l1_indexing
  "percent": 45,
  "current": 45,
  "total": 100,
  "details": "处理中: file1.pdf",
  "timestamp": "2025-12-13T10:00:02"
}
```

### 4. 状态更新
```json
{
  "type": "status",
  "status": "running",  // started | running | completed | failed | stopped
  "message": "图谱构建任务正在运行",
  "timestamp": "2025-12-13T10:00:03"
}
```

### 5. 文件状态
```json
{
  "type": "file_status",
  "file_path": "/path/to/file.pdf",
  "status": "processing",  // processing | completed | failed | queued
  "stage": "L0",  // L0 | L1
  "progress": 50,
  "timestamp": "2025-12-13T10:00:04"
}
```

### 6. 统计信息
```json
{
  "type": "stats",
  "data": {
    "l0_files": 10,
    "l1_tasks": 8,
    "entities": 1234,
    "relations": 567
  },
  "timestamp": "2025-12-13T10:00:05"
}
```

### 7. 错误消息
```json
{
  "type": "error",
  "error": "实体提取失败",
  "details": {
    "file": "file1.pdf",
    "reason": "解析错误"
  },
  "timestamp": "2025-12-13T10:00:06"
}
```

## 🔧 HTTP API 端点

### 1. 触发构建
```bash
POST /build/run
Content-Type: application/json

{
  "mode": "incremental",  // incremental | full
  "force": false,
  "skip_l0": false,
  "skip_l1": false,
  "file_paths": null  // null 表示处理所有变更文件
}
```

**响应**：
```json
{
  "status": "started",
  "message": "构建任务已在后台启动",
  "task_id": "build_001"
}
```

### 2. 停止构建
```bash
POST /build/stop
```

**响应**：
```json
{
  "status": "stopped",
  "message": "构建任务停止指令已发送"
}
```

### 3. 查询状态
```bash
GET /build/status
```

**响应**：
```json
{
  "is_running": true,
  "active_connections": 2
}
```

## 🎨 自定义样式

所有示例都使用内联样式，便于复制和修改。你可以：

1. **修改颜色主题**：
   - 主色：`#3b82f6` (蓝色)
   - 成功：`#10b981` (绿色)
   - 警告：`#fbbf24` (黄色)
   - 错误：`#ef4444` (红色)

2. **调整布局**：
   - 容器宽度：`max-width: 900px`
   - 卡片阴影：`box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2)`
   - 边框圆角：`border-radius: 16px`

3. **修改进度条样式**：
   - 高度：`height: 24px`
   - 渐变色：`linear-gradient(90deg, #3b82f6, #8b5cf6)`
   - 动画：`transition: width 0.3s ease`

## 🐛 常见问题

### Q1: WebSocket 连接失败？
- **检查后端服务**：确保 `python server/main.py` 正在运行
- **检查端口**：确认后端运行在 `http://localhost:8000`
- **跨域问题**：如果前端和后端不在同一域名，需要配置 CORS

### Q2: 进度条不更新？
- **检查 WebSocket 连接**：确保连接状态显示「已连接」
- **查看浏览器控制台**：检查是否有 JavaScript 错误
- **检查后端日志**：确认后端正在发送消息

### Q3: 如何在生产环境使用？
- **修改 API URL**：将 `localhost:8000` 替换为生产环境地址
- **使用 WSS**：生产环境应使用 `wss://` 而不是 `ws://`
- **添加认证**：在 WebSocket 连接时添加 token 验证

### Q4: 如何添加更多统计信息？
1. 修改后端 `emit_stats` 方法，添加新字段
2. 修改前端 `handleMessage` 中的 `stats` 更新逻辑
3. 在 HTML 中添加新的统计卡片

## 📚 参考资料

- [WebSocket API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [React Hooks](https://react.dev/reference/react)
- [Vue 3 Composition API](https://vuejs.org/guide/extras/composition-api-faq.html)

## 💡 进阶功能

如果需要更高级的功能，可以考虑：

1. **断线重连**：WebSocket 断开后自动重连
2. **消息队列**：缓存离线期间的消息
3. **多任务管理**：同时监控多个构建任务
4. **历史记录**：保存构建历史和日志
5. **通知系统**：构建完成后桌面通知
6. **性能图表**：使用 Chart.js 绘制性能趋势图

## 📝 许可证

MIT License

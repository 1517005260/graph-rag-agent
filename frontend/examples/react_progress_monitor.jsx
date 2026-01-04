/**
 * React 图谱构建进度监控组件
 *
 * 使用方法：
 * import BuildProgressMonitor from './react_progress_monitor';
 *
 * function App() {
 *   return <BuildProgressMonitor apiUrl="http://localhost:8000" />;
 * }
 */

import React, { useState, useEffect, useRef } from 'react';

const BuildProgressMonitor = ({ apiUrl = 'http://localhost:8000' }) => {
  // WebSocket 状态
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  // 进度状态
  const [l0Progress, setL0Progress] = useState({ percent: 0, details: '' });
  const [l1Progress, setL1Progress] = useState({ percent: 0, details: '' });

  // 日志
  const [logs, setLogs] = useState([]);

  // 统计数据
  const [stats, setStats] = useState({
    l0Files: 0,
    l1Tasks: 0,
    entities: 0,
    relations: 0
  });

  // 连接 WebSocket
  const connectWebSocket = () => {
    const wsUrl = apiUrl.replace('http', 'ws') + '/build/ws/progress';

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      addLog('WebSocket 已连接', 'INFO');
      return;
    }

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setConnected(true);
      addLog('✅ WebSocket 连接成功', 'INFO');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleMessage(data);
    };

    ws.onerror = (error) => {
      console.error('WebSocket 错误:', error);
      addLog('❌ WebSocket 连接错误', 'ERROR');
    };

    ws.onclose = () => {
      setConnected(false);
      addLog('WebSocket 连接已关闭', 'WARNING');
    };

    wsRef.current = ws;
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
          setL0Progress({ percent: data.percent, details: data.details || '' });
        } else if (data.stage === 'l1_indexing') {
          setL1Progress({ percent: data.percent, details: data.details || '' });
        }
        break;

      case 'status':
        addLog(`状态更新: ${data.status} - ${data.message}`, 'INFO');
        break;

      case 'file_status':
        addLog(`文件 ${data.file_path}: ${data.status} (${data.stage})`, 'INFO');
        break;

      case 'stats':
        setStats({
          l0Files: data.data.l0_files || 0,
          l1Tasks: data.data.l1_tasks || 0,
          entities: data.data.entities || 0,
          relations: data.data.relations || 0
        });
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
      id: Date.now(),
      timestamp,
      message,
      level
    };

    setLogs(prevLogs => [logEntry, ...prevLogs.slice(0, 49)]); // 保留最新 50 条
  };

  // 触发构建
  const startBuild = async () => {
    try {
      const response = await fetch(`${apiUrl}/build/run`, {
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
      const response = await fetch(`${apiUrl}/build/stop`, { method: 'POST' });
      const result = await response.json();
      addLog(`停止构建: ${result.message}`, 'WARNING');
    } catch (error) {
      addLog(`停止构建失败: ${error.message}`, 'ERROR');
    }
  };

  // 组件卸载时关闭 WebSocket
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🚀 图谱构建进度监控</h1>
      <p style={styles.subtitle}>WebSocket 实时推送 (React Demo)</p>

      {/* 连接状态 */}
      <div style={styles.statusBadge(connected)}>
        {connected ? '已连接' : '未连接'}
      </div>

      {/* 控制按钮 */}
      <div style={styles.controls}>
        <button onClick={connectWebSocket} style={styles.button('#3b82f6')}>
          连接 WebSocket
        </button>
        <button onClick={startBuild} style={styles.button('#10b981')}>
          触发构建
        </button>
        <button onClick={stopBuild} style={styles.button('#ef4444')}>
          停止构建
        </button>
      </div>

      {/* 统计卡片 */}
      <div style={styles.statsGrid}>
        <StatCard value={stats.l0Files} label="L0 处理文件数" />
        <StatCard value={stats.l1Tasks} label="L1 提交任务数" />
        <StatCard value={stats.entities} label="实体数量" />
        <StatCard value={stats.relations} label="关系数量" />
      </div>

      {/* 进度条 */}
      <div style={styles.progressSection}>
        <ProgressBar
          label="L0 快速索引"
          percent={l0Progress.percent}
          details={l0Progress.details}
        />
        <ProgressBar
          label="L1 深度索引"
          percent={l1Progress.percent}
          details={l1Progress.details}
        />
      </div>

      {/* 日志区域 */}
      <div style={styles.logsSection}>
        <div style={styles.logsTitle}>📋 实时日志</div>
        <div style={styles.logsContainer}>
          {logs.map(log => (
            <div key={log.id} style={styles.logEntry(log.level)}>
              [{log.timestamp}] {log.message}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// 统计卡片组件
const StatCard = ({ value, label }) => (
  <div style={styles.statCard}>
    <div style={styles.statValue}>{value}</div>
    <div style={styles.statLabel}>{label}</div>
  </div>
);

// 进度条组件
const ProgressBar = ({ label, percent, details }) => (
  <div style={styles.progressItem}>
    <div style={styles.progressLabel}>
      <span>{details || label}</span>
      <span>{percent}%</span>
    </div>
    <div style={styles.progressBarContainer}>
      <div style={styles.progressBar(percent)}>{percent}%</div>
    </div>
  </div>
);

// 样式
const styles = {
  container: {
    maxWidth: '900px',
    margin: '0 auto',
    background: 'white',
    borderRadius: '16px',
    boxShadow: '0 10px 40px rgba(0, 0, 0, 0.2)',
    padding: '30px'
  },
  title: {
    color: '#333',
    marginBottom: '10px',
    fontSize: '28px'
  },
  subtitle: {
    color: '#666',
    marginBottom: '30px',
    fontSize: '14px'
  },
  statusBadge: (connected) => ({
    display: 'inline-block',
    padding: '6px 12px',
    borderRadius: '20px',
    fontSize: '12px',
    fontWeight: '600',
    marginBottom: '20px',
    background: connected ? '#10b981' : '#ef4444',
    color: 'white'
  }),
  controls: {
    display: 'flex',
    gap: '10px',
    marginBottom: '30px'
  },
  button: (color) => ({
    padding: '12px 24px',
    border: 'none',
    borderRadius: '8px',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    background: color,
    color: 'white',
    transition: 'all 0.3s'
  }),
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
    gap: '15px',
    marginBottom: '30px'
  },
  statCard: {
    background: '#f3f4f6',
    padding: '15px',
    borderRadius: '8px',
    textAlign: 'center'
  },
  statValue: {
    fontSize: '24px',
    fontWeight: '700',
    color: '#3b82f6',
    marginBottom: '5px'
  },
  statLabel: {
    fontSize: '12px',
    color: '#6b7280'
  },
  progressSection: {
    marginBottom: '30px'
  },
  progressItem: {
    marginBottom: '20px'
  },
  progressLabel: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '8px',
    fontSize: '14px',
    color: '#333'
  },
  progressBarContainer: {
    width: '100%',
    height: '24px',
    background: '#e5e7eb',
    borderRadius: '12px',
    overflow: 'hidden'
  },
  progressBar: (percent) => ({
    width: `${percent}%`,
    height: '100%',
    background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
    transition: 'width 0.3s ease',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontSize: '12px',
    fontWeight: '600'
  }),
  logsSection: {
    background: '#f9fafb',
    borderRadius: '8px',
    padding: '20px'
  },
  logsTitle: {
    fontSize: '16px',
    fontWeight: '600',
    marginBottom: '12px',
    color: '#333'
  },
  logsContainer: {
    maxHeight: '400px',
    overflowY: 'auto'
  },
  logEntry: (level) => ({
    padding: '8px 12px',
    marginBottom: '6px',
    borderRadius: '6px',
    fontSize: '13px',
    fontFamily: "'Courier New', monospace",
    background: level === 'ERROR' ? '#fee2e2' : level === 'WARNING' ? '#fef3c7' : '#dbeafe',
    color: level === 'ERROR' ? '#991b1b' : level === 'WARNING' ? '#92400e' : '#1e40af'
  })
};

export default BuildProgressMonitor;

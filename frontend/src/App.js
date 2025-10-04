import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Layout, Menu, Card, Row, Col, Statistic, Badge, notification, Spin } from 'antd';
import { 
  HomeOutlined, 
  ExperimentOutlined, 
  HistoryOutlined, 
  BarChartOutlined,
  SettingOutlined,
  SafetyCertificateOutlined,
  ApiOutlined,
  RobotOutlined
} from '@ant-design/icons';
import axios from 'axios';
import { SecurityScanOutlined, SafetyOutlined } from '@ant-design/icons';

// 導入頁面組件
import RealTimeTest from './components/RealTimeTest';
import TestHistory from './components/TestHistory';
import BatchAnalysis from './components/BatchAnalysis';
import Settings from './components/Settings';
import Dashboard from './components/Dashboard';
import MultiModelTest from './components/MultiModelTest';

import './App.css';

const { Header, Content, Sider } = Layout;

// 配置 axios 默認設置
axios.defaults.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:5001';
axios.defaults.timeout = 60000;

// 添加請求攔截器
axios.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 500) {
      notification.error({
        message: 'Server Error',
        description: 'Internal server error occurred. Please try again.',
        duration: 4
      });
    } else if (error.code === 'ECONNABORTED') {
      notification.error({
        message: 'Request Timeout',
        description: 'Request took too long to complete. Please try again.',
        duration: 4
      });
    }
    return Promise.reject(error);
  }
);

function App() {
  const [systemStatus, setSystemStatus] = useState({
    model_exists: false,
    ollama_connected: false,
    defense_system_ready: false,
    available_models: [],
    config: {}
  });
  const [loading, setLoading] = useState(true);
  const [selectedMenu, setSelectedMenu] = useState('dashboard');

  // 獲取系統狀態
  const fetchSystemStatus = async () => {
    try {
      const response = await axios.get('/api/status');
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
      notification.error({
        message: 'System Status Error',
        description: 'Failed to get system status. Please check if the backend server is running.',
        duration: 5
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    // 定期檢查系統狀態
    const interval = setInterval(fetchSystemStatus, 60000);
    return () => clearInterval(interval);
  }, []);

  const menuItems = [
    {
      key: 'dashboard',
      icon: <HomeOutlined />,
      label: 'Dashboard',
      path: '/'
    },
    {
      key: 'realtime',
      icon: <ExperimentOutlined />,
      label: 'Real-time Testing',
      path: '/realtime'
    },
    {
      key: 'history',
      icon: <HistoryOutlined />,
      label: 'Test History',
      path: '/history'
    },
    {
      key: 'batch',
      icon: <BarChartOutlined />,
      label: 'Batch Analysis',
      path: '/batch'
    },
    {
      key: 'multimodel',           // 新增
      icon: <RobotOutlined />,      // 新增（或用其他圖示）
      label: 'Multi-Model Test',    // 新增
      path: '/multimodel'           // 新增
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Settings',
      path: '/settings'
    }
  ];

const getStatusBadge = (status, label) => {
  const color = status ? 'success' : 'error';
  const text = status ? 'Connected' : 'Disconnected';
  return (
    <Badge 
      status={color} 
      text={`${label}: ${text}`}
      style={{ color: '#fff' }}  // ✅ 加上這行
    />
  );
};

  const getSystemHealthScore = () => {
    const checks = [
      systemStatus.model_exists,
      systemStatus.ollama_connected,
      systemStatus.defense_system_ready
    ];
    const score = (checks.filter(Boolean).length / checks.length) * 100;
    return Math.round(score);
  };

  if (loading) {
    return (
      <Layout style={{ minHeight: '100vh' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh',
          flexDirection: 'column'
        }}>
          <Spin size="large" />
          <p style={{ marginTop: 16, fontSize: 16 }}>Loading AI Defense System...</p>
        </div>
      </Layout>
    );
  }

  return (
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ 
          padding: '0 24px', 
          background: '#001529',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ 
            color: 'white', 
            fontSize: '20px', 
            fontWeight: 'bold',
            display: 'flex',
            alignItems: 'center'
          }}>
            <SafetyCertificateOutlined style={{ marginRight: 8, fontSize: 24 }} />
            AI Jailbreak Defense System
          </div>
          
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            {getStatusBadge(systemStatus.model_exists, 'Model')}
            {getStatusBadge(systemStatus.ollama_connected, 'Ollama')}
            {getStatusBadge(systemStatus.defense_system_ready, 'Defense')}
          </div>
        </Header>

        <Layout>
          <Sider 
            width={200} 
            style={{ background: '#fff' }}
            breakpoint="lg"
            collapsedWidth="0"
          >
            <Menu
              mode="inline"
              selectedKeys={[selectedMenu]}
              style={{ height: '100%', borderRight: 0 }}
              items={menuItems.map(item => ({
                key: item.key,
                icon: item.icon,
                label: (
                  <Link 
                    to={item.path}
                    onClick={() => setSelectedMenu(item.key)}
                  >
                    {item.label}
                  </Link>
                )
              }))}
            />
          </Sider>

          <Layout style={{ padding: '24px' }}>
            <Content style={{ 
              background: '#fff', 
              padding: 24, 
              margin: 0, 
              minHeight: 280,
              borderRadius: 8,
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
            }}>
              {/* 系統狀態卡片 */}
              <Row gutter={16} style={{ marginBottom: 24 }}>
                <Col span={6}>
                  <Card>
                    <Statistic
                      title="System Health"
                      value={getSystemHealthScore()}
                      suffix="%"
                      valueStyle={{ 
                        color: getSystemHealthScore() > 80 ? '#3f8600' : 
                               getSystemHealthScore() > 50 ? '#faad14' : '#cf1322'
                      }}
                      prefix={<ApiOutlined />}
                    />
                  </Card>
                </Col>
                <Col span={6}>
                  <Card>
                    <Statistic
                      title="Available Models"
                      value={systemStatus.available_models?.length || 0}
                      prefix={<RobotOutlined />}
                    />
                  </Card>
                </Col>
                <Col span={6}>
                  <Card>
                    <Statistic
                      title="Defense Status"
                      value={systemStatus.defense_system_ready ? 'Active' : 'Inactive'}
                      valueStyle={{ 
                        color: systemStatus.defense_system_ready ? '#3f8600' : '#cf1322'
                      }}
                      prefix={<SafetyCertificateOutlined />}
                    />
                  </Card>
                </Col>
                <Col span={6}>
                  <Card>
                    <Statistic
                      title="Ollama Service"
                      value={systemStatus.ollama_connected ? 'Online' : 'Offline'}
                      valueStyle={{ 
                        color: systemStatus.ollama_connected ? '#3f8600' : '#cf1322'
                      }}
                      prefix={<ApiOutlined />}
                    />
                  </Card>
                </Col>
              </Row>

              {/* 路由內容 */}
              <Routes>
                <Route 
                  path="/" 
                  element={
                    <Dashboard 
                      systemStatus={systemStatus} 
                      onRefresh={fetchSystemStatus}
                    />
                  } 
                />
                <Route 
                  path="/realtime" 
                  element={
                    <RealTimeTest 
                      systemStatus={systemStatus}
                      onRefresh={fetchSystemStatus}
                    />
                  } 
                />
                <Route 
                  path="/history" 
                  element={<TestHistory />} 
                />
                <Route 
                  path="/batch" 
                  element={<BatchAnalysis />} 
                />
                 {/* 新增這個路由 */}
                <Route 
                  path="/multimodel" 
                  element={<MultiModelTest />} 
                />
                <Route 
                  path="/settings" 
                  element={
                    <Settings 
                      systemStatus={systemStatus}
                      onUpdate={fetchSystemStatus}
                    />
                  } 
                />
              </Routes>
            </Content>
          </Layout>
        </Layout>
      </Layout>
    </Router>
  );
}

export default App;
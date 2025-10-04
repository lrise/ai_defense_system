import React from 'react';
import { Card, Row, Col, Alert, Typography, Timeline, Tag } from 'antd';
import { 
  SafetyCertificateOutlined, 
  RobotOutlined, 
  ExperimentOutlined,
  BarChartOutlined
} from '@ant-design/icons';

const { Title, Paragraph, Text } = Typography;

const Dashboard = ({ systemStatus, onRefresh }) => {
  const getSystemHealthScore = () => {
    const checks = [
      systemStatus.model_exists,
      systemStatus.ollama_connected,
      systemStatus.defense_system_ready
    ];
    return (checks.filter(Boolean).length / checks.length) * 100;
  };

  const healthScore = getSystemHealthScore();

  return (
    <div>
      <Title level={2}>System Dashboard</Title>
      
      <Row gutter={24} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Alert
            message="AI Jailbreak Defense System"
            description="A comprehensive defense system to detect and prevent AI jailbreak attacks with intelligent alternative response generation."
            type="info"
            showIcon
            style={{ marginBottom: 24 }}
          />
        </Col>
      </Row>

      <Row gutter={24}>
        <Col span={12}>
          <Card title="Quick Start Guide">
            <Timeline>
              <Timeline.Item 
                dot={<ExperimentOutlined style={{ fontSize: '16px' }} />}
                color={systemStatus.defense_system_ready ? 'green' : 'red'}
              >
                <Text strong>Real-time Testing</Text>
                <br />
                <Text type="secondary">
                  Test AI responses in auto-generation or manual input mode
                </Text>
              </Timeline.Item>
              <Timeline.Item 
                dot={<BarChartOutlined style={{ fontSize: '16px' }} />}
                color="blue"
              >
                <Text strong>Batch Analysis</Text>
                <br />
                <Text type="secondary">
                  Upload CSV datasets for comprehensive vulnerability analysis
                </Text>
              </Timeline.Item>
              <Timeline.Item 
                dot={<SafetyCertificateOutlined style={{ fontSize: '16px' }} />}
                color="orange"
              >
                <Text strong>View Results</Text>
                <br />
                <Text type="secondary">
                  Monitor test history and download analysis reports
                </Text>
              </Timeline.Item>
            </Timeline>
          </Card>

          <Card title="System Features" style={{ marginTop: 24 }}>
            <Row gutter={16}>
              <Col span={12}>
                <ul>
                  <li>Multi-layer defense detection</li>
                  <li>Context-aware analysis</li>
                  <li>Real-time safety assessment</li>
                  <li>Intelligent response generation</li>
                </ul>
              </Col>
              <Col span={12}>
                <ul>
                  <li>Batch dataset processing</li>
                  <li>Model vulnerability comparison</li>
                  <li>Comprehensive reporting</li>
                  <li>Privacy-first local processing</li>
                </ul>
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={12}>
          <Card title={`System Health (${healthScore.toFixed(0)}%)`}>
            <Row gutter={16}>
              <Col span={24}>
                <div style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Text>Toxicity Detection Model</Text>
                    <Tag color={systemStatus.model_exists ? 'success' : 'error'}>
                      {systemStatus.model_exists ? 'Available' : 'Missing'}
                    </Tag>
                  </div>
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Text>Ollama Service</Text>
                    <Tag color={systemStatus.ollama_connected ? 'success' : 'error'}>
                      {systemStatus.ollama_connected ? 'Connected' : 'Disconnected'}
                    </Tag>
                  </div>
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Text>Defense System</Text>
                    <Tag color={systemStatus.defense_system_ready ? 'success' : 'error'}>
                      {systemStatus.defense_system_ready ? 'Ready' : 'Not Ready'}
                    </Tag>
                  </div>
                </div>

                {systemStatus.available_models?.length > 0 && (
                  <div>
                    <Text strong>Available Models ({systemStatus.available_models.length})</Text>
                    <div style={{ marginTop: 8, maxHeight: 200, overflowY: 'auto' }}>
                      {systemStatus.available_models.map(model => (
                        <Tag key={model} style={{ marginBottom: 4, display: 'block' }}>
                          <RobotOutlined /> {model}
                        </Tag>
                      ))}
                    </div>
                  </div>
                )}
              </Col>
            </Row>
          </Card>

          <Card title="System Requirements" style={{ marginTop: 24 }}>
            <Paragraph>
              <Text strong>Backend Requirements:</Text>
              <ul>
                <li>Python 3.8+</li>
                <li>Flask and required dependencies</li>
                <li>Toxicity detection model</li>
                <li>Ollama service running locally</li>
              </ul>
              
              <Text strong>Recommended Setup:</Text>
              <ul>
                <li>8GB+ RAM for model processing</li>
                <li>GPU acceleration (optional)</li>
                <li>SSD storage for faster model loading</li>
              </ul>
            </Paragraph>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
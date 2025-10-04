// src/components/MultiModelTest.js
import React, { useState } from 'react';
import { 
  Card, 
  Input, 
  Button, 
  Row, 
  Col, 
  Tag, 
  Alert, 
  Spin, 
  Typography,
  Statistic,
  Space,
  Divider,
  Progress
} from 'antd';
import { 
  SendOutlined, 
  RobotOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined
} from '@ant-design/icons';
import axios from 'axios';

const { TextArea } = Input;
const { Title, Text, Paragraph } = Typography;

const MultiModelTest = () => {
  const [prompt, setPrompt] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [testComplete, setTestComplete] = useState(false);

  const handleTest = async () => {
  if (!prompt.trim()) {
    return;
  }

  setLoading(true);
  setTestComplete(false);
  setResults([]);

  try {
    // 明確指定完整 URL
    const response = await axios.post('http://localhost:5001/api/multi-model-test', {
      prompt: prompt.trim() },
      { timeout: 180000 } 
    );

    console.log('Response:', response.data); // 加這行看看回傳什麼

    setResults(response.data.results);
    setTestComplete(true);
  } catch (error) {
    console.error('Multi-model test failed:', error);
    console.error('Error details:', error.response); // 加這行看詳細錯誤
    
    // 顯示錯誤訊息給用戶
    alert(`Error: ${error.response?.data?.error || error.message}`);
  } finally {
    setLoading(false);
  }
};
  const getRiskColor = (riskLevel) => {
    const colors = {
      'safe': 'success',
      'low': 'processing',
      'medium': 'warning',
      'high': 'error',
      'critical': 'error'
    };
    return colors[riskLevel] || 'default';
  };

  const getRiskIcon = (isBlocked) => {
    return isBlocked ? 
      <CloseCircleOutlined style={{ color: '#ff4d4f' }} /> : 
      <CheckCircleOutlined style={{ color: '#52c41a' }} />;
  };

  const getStatistics = () => {
    if (results.length === 0) return null;

    const successful = results.filter(r => r.response).length;
    const blocked = results.filter(r => r.is_blocked).length;
    const safeCount = results.filter(r => r.risk_level === 'safe' || r.risk_level === 'low').length;

    return (
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Models Tested"
              value={successful}
              suffix={`/ ${results.length}`}
              prefix={<RobotOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Blocked Responses"
              value={blocked}
              valueStyle={{ color: blocked > 0 ? '#cf1322' : '#3f8600' }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Safe Responses"
              value={safeCount}
              valueStyle={{ color: '#3f8600' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Safety Rate"
              value={successful > 0 ? Math.round((safeCount / successful) * 100) : 0}
              suffix="%"
              valueStyle={{ 
                color: (safeCount / successful) > 0.7 ? '#3f8600' : '#faad14'
              }}
            />
          </Card>
        </Col>
      </Row>
    );
  };

  return (
    <div>
      <Title level={2}>
        <RobotOutlined /> Multi-Model Safety Comparison
      </Title>
      <Paragraph type="secondary">
        Test the same prompt across multiple LLMs and compare their safety responses
      </Paragraph>

      <Card style={{ marginBottom: 24 }}>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <div>
            <Text strong>Test Prompt:</Text>
            <TextArea
              rows={4}
              placeholder="Enter a prompt to test across all models..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={loading}
            />
          </div>

          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleTest}
            loading={loading}
            disabled={!prompt.trim()}
            size="large"
          >
            {loading ? 'Testing Models...' : 'Test All Models'}
          </Button>

          {loading && (
            <Alert
              message="Running safety tests..."
              description="Testing your prompt across multiple models. This may take 30-60 seconds."
              type="info"
              showIcon
            />
          )}
        </Space>
      </Card>

      {testComplete && getStatistics()}

      <Row gutter={[16, 16]}>
        {results.map((result, index) => (
          <Col xs={24} sm={24} md={12} lg={8} xl={8} key={index}>
            <Card
              title={
                <Space>
                  <RobotOutlined />
                  <Text strong>{result.model}</Text>
                </Space>
              }
              extra={
                result.error ? (
                  <Tag color="error">Error</Tag>
                ) : (
                  <Tag color={getRiskColor(result.risk_level)}>
                    {getRiskIcon(result.is_blocked)}
                    {' '}
                    {result.risk_level?.toUpperCase()}
                  </Tag>
                )
              }
              hoverable
            >
              {result.error ? (
                <Alert
                  message="Model Error"
                  description={result.error}
                  type="error"
                  showIcon
                />
              ) : (
                <>
                  <div style={{ 
                    background: '#f5f5f5', 
                    padding: 12, 
                    borderRadius: 4,
                    marginBottom: 12,
                    maxHeight: 150,
                    overflowY: 'auto'
                  }}>
                    <Text style={{ fontSize: '0.9em' }}>
                      {result.response}
                    </Text>
                  </div>

                  <Space direction="vertical" style={{ width: '100%' }} size="small">
                    <div>
                      <Text type="secondary" style={{ fontSize: '0.85em' }}>
                        Confidence: 
                      </Text>
                      <Progress 
                        percent={Math.round(result.confidence * 100)} 
                        size="small"
                        status={result.is_blocked ? 'exception' : 'success'}
                        style={{ marginBottom: 0 }}
                      />
                    </div>

                    {result.triggered_rules && result.triggered_rules.length > 0 && (
                      <div>
                        <Text type="secondary" style={{ fontSize: '0.85em' }}>
                          Triggered Rules:
                        </Text>
                        <div style={{ marginTop: 4 }}>
                          {result.triggered_rules.map((rule, idx) => (
                            <Tag 
                              key={idx} 
                              color="orange" 
                              style={{ fontSize: '0.75em', marginBottom: 4 }}
                            >
                              {rule}
                            </Tag>
                          ))}
                        </div>
                      </div>
                    )}

                    {result.detailed_scores && (
                      <div>
                        <Text type="secondary" style={{ fontSize: '0.85em' }}>
                          Scores:
                        </Text>
                        <div style={{ marginTop: 4 }}>
                          {result.detailed_scores.toxicity !== undefined && (
                            <Tag color="blue" style={{ fontSize: '0.75em' }}>
                              Toxicity: {(result.detailed_scores.toxicity * 100).toFixed(1)}%
                            </Tag>
                          )}
                          {result.detailed_scores.llm_harmful !== undefined && (
                            <Tag color="purple" style={{ fontSize: '0.75em' }}>
                              Harmful: {(result.detailed_scores.llm_harmful * 100).toFixed(0)}%
                            </Tag>
                          )}
                        </div>
                      </div>
                    )}
                  </Space>
                </>
              )}
            </Card>
          </Col>
        ))}
      </Row>

      {results.length === 0 && !loading && (
        <Alert
          message="No Results Yet"
          description="Enter a prompt above and click 'Test All Models' to begin the safety comparison."
          type="info"
          showIcon
        />
      )}
    </div>
  );
};

export default MultiModelTest;
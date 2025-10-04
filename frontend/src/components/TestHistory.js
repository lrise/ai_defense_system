import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Table, 
  Button, 
  Select, 
  Row, 
  Col, 
  Statistic, 
  Tag,
  notification,
  Space,
  Typography
} from 'antd';
import { DeleteOutlined, DownloadOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Option } = Select;
const { Title } = Typography;

const TestHistory = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [riskFilter, setRiskFilter] = useState('all');
  const [blockedFilter, setBlockedFilter] = useState('all');

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (riskFilter !== 'all') params.append('risk_level', riskFilter);
      if (blockedFilter !== 'all') params.append('blocked', blockedFilter);
      
      const response = await axios.get(`/api/history?${params}`);
      setHistory(response.data);
    } catch (error) {
      notification.error({
        message: 'Failed to Load History',
        description: 'Could not retrieve test history'
      });
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = async () => {
    try {
      await axios.delete('/api/history');
      setHistory([]);
      notification.success({
        message: 'History Cleared',
        description: 'All test history has been cleared'
      });
    } catch (error) {
      notification.error({
        message: 'Failed to Clear History',
        description: 'Could not clear test history'
      });
    }
  };

  const downloadHistory = () => {
    const csvContent = [
      ['Timestamp', 'Prompt', 'Response', 'Test Mode', 'Test Model', 'Risk Level', 'Toxicity Score', 'Blocked'],
      ...history.map(item => [
        item.timestamp,
        `"${item.prompt.replace(/"/g, '""')}"`,
        `"${item.response.replace(/"/g, '""')}"`,
        item.test_mode,
        item.test_model,
        item.risk_level,
        item.toxicity_score,
        item.blocked
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `test_history_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  useEffect(() => {
    fetchHistory();
  }, [riskFilter, blockedFilter]);

  const columns = [
    {
      title: 'Timestamp',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
    },
    {
      title: 'Prompt',
      dataIndex: 'prompt',
      key: 'prompt',
      ellipsis: true,
      width: 200,
    },
    {
      title: 'Response',
      dataIndex: 'response', 
      key: 'response',
      ellipsis: true,
      width: 200,
    },
    {
      title: 'Test Mode',
      dataIndex: 'test_mode',
      key: 'test_mode',
      width: 120,
    },
    {
      title: 'Model',
      dataIndex: 'test_model',
      key: 'test_model',
      width: 120,
    },
    {
      title: 'Risk Level',
      dataIndex: 'risk_level',
      key: 'risk_level',
      render: (level) => {
        const colors = {
          'safe': 'green',
          'low': 'blue',
          'medium': 'orange', 
          'high': 'red',
          'critical': 'red'
        };
        return <Tag color={colors[level]}>{level?.toUpperCase()}</Tag>;
      }
    },
    {
      title: 'Toxicity',
      dataIndex: 'toxicity_score',
      key: 'toxicity_score',
      width: 100,
    },
    {
      title: 'Status',
      dataIndex: 'blocked',
      key: 'blocked',
      render: (blocked) => (
        <Tag color={blocked ? 'red' : 'green'}>
          {blocked ? 'BLOCKED' : 'PASSED'}
        </Tag>
      )
    }
  ];

  const stats = {
    total: history.length,
    blocked: history.filter(h => h.blocked).length,
    avgToxicity: history.length > 0 ? 
      (history.reduce((sum, h) => sum + parseFloat(h.toxicity_score || 0), 0) / history.length).toFixed(3) : 0,
    highRisk: history.filter(h => ['high', 'critical'].includes(h.risk_level)).length
  };

  return (
    <div>
      <Title level={2}>📋 Test History Records</Title>
      
      <Card style={{ marginBottom: 24 }}>
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Statistic title="Total Tests" value={stats.total} />
          </Col>
          <Col span={6}>
            <Statistic 
              title="Defense Trigger Rate" 
              value={stats.total > 0 ? ((stats.blocked / stats.total) * 100).toFixed(1) : 0}
              suffix="%" 
            />
          </Col>
          <Col span={6}>
            <Statistic title="Average Toxicity" value={stats.avgToxicity} />
          </Col>
          <Col span={6}>
            <Statistic title="High Risk Cases" value={stats.highRisk} />
          </Col>
        </Row>

        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Select
              placeholder="Filter by Risk Level"
              value={riskFilter}
              onChange={setRiskFilter}
              style={{ width: '100%' }}
            >
              <Option value="all">All Risk Levels</Option>
              <Option value="safe">Safe</Option>
              <Option value="low">Low</Option>
              <Option value="medium">Medium</Option>
              <Option value="high">High</Option>
              <Option value="critical">Critical</Option>
            </Select>
          </Col>
          <Col span={6}>
            <Select
              placeholder="Filter by Defense Status"
              value={blockedFilter}
              onChange={setBlockedFilter}
              style={{ width: '100%' }}
            >
              <Option value="all">All Statuses</Option>
              <Option value="true">Blocked</Option>
              <Option value="false">Passed</Option>
            </Select>
          </Col>
          <Col span={12}>
            <Space>
              <Button 
                type="primary"
                icon={<DownloadOutlined />}
                onClick={downloadHistory}
                disabled={history.length === 0}
              >
                Download History
              </Button>
              <Button 
                danger
                icon={<DeleteOutlined />}
                onClick={clearHistory}
                disabled={history.length === 0}
              >
                Clear History
              </Button>
            </Space>
          </Col>
        </Row>

        <Table
          dataSource={history}
          columns={columns}
          loading={loading}
          rowKey="id"
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} items`
          }}
          scroll={{ x: 1000 }}
        />
      </Card>
    </div>
  );
};

export default TestHistory;
import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Upload, 
  Button, 
  Progress, 
  Table, 
  Row, 
  Col, 
  Statistic, 
  Select, 
  Checkbox,
  Alert,
  Modal,
  Tag,
  Divider,
  notification,
  Typography,
  Space,
  Tabs,
  Radio,
  InputNumber
} from 'antd';
import { 
  UploadOutlined, 
  PlayCircleOutlined, 
  DownloadOutlined,
  FileExcelOutlined,
  BarChartOutlined,
  TrophyOutlined,
  ShieldCheckOutlined
} from '@ant-design/icons';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter
} from 'recharts';
import axios from 'axios';

const { Dragger } = Upload;
const { Option } = Select;
const { Title, Text } = Typography;
const { TabPane } = Tabs;

const BatchAnalysis = () => {
  const [uploadedData, setUploadedData] = useState(null);
  const [analysisId, setAnalysisId] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
  const [sampleSize, setSampleSize] = useState('all');
  const [generateAlternatives, setGenerateAlternatives] = useState(false);
  
  // 新增的狀態
  const [customSampleSize, setCustomSampleSize] = useState('');
  const [sampleSizeMode, setSampleSizeMode] = useState('preset'); // 'preset' or 'custom'

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: '.csv',
    beforeUpload: (file) => {
      if (!file.name.toLowerCase().endsWith('.csv')) {
        notification.error({
          message: 'Invalid File Type',
          description: 'Please upload a CSV file.'
        });
        return false;
      }
      return false; // Prevent auto upload
    },
    onChange: async (info) => {
      const { file } = info;
      if (file.status !== 'error') {
        const formData = new FormData();
        formData.append('file', file.originFileObj || file);
        
        try {
          const response = await axios.post('/api/batch-upload', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });
          
          setUploadedData(response.data);
          setAnalysisId(response.data.analysis_id);
          setSelectedModels([]); // Reset selections
          
          notification.success({
            message: 'File Uploaded Successfully',
            description: `${response.data.total_records} records loaded.`
          });
        } catch (error) {
          notification.error({
            message: 'Upload Failed',
            description: error.response?.data?.error || 'Failed to upload file'
          });
        }
      }
    },
    showUploadList: false,
  };

  const startAnalysis = async () => {
    if (!analysisId) {
      notification.warning({
        message: '沒有數據',
        description: '請先上傳數據集。'
      });
      return;
    }

    if (selectedModels.length === 0) {
      notification.warning({
        message: '未選擇模型',
        description: '請至少選擇一個模型進行分析。'
      });
      return;
    }

    setAnalyzing(true);
    setAnalysisProgress(0);
    setAnalysisResults(null);

    try {
      console.log(`開始分析，分析ID: ${analysisId}`);
      console.log('選中的模型:', selectedModels);
      console.log('樣本大小模式:', sampleSizeMode);
      console.log('樣本大小:', sampleSizeMode === 'custom' ? customSampleSize : sampleSize);

      // 決定最終的樣本大小
      const finalSampleSize = sampleSizeMode === 'custom' ? 
        customSampleSize : 
        (sampleSize === 'all' ? 'all' : parseInt(sampleSize));

      await axios.post(`/api/batch-analyze/${analysisId}`, {
        selected_models: selectedModels,
        sample_size: finalSampleSize,
        generate_alternatives: generateAlternatives
      });

      // 開始輪詢進度
      const pollInterval = setInterval(async () => {
        try {
          console.log(`輪詢狀態，分析ID: ${analysisId}`);
          
          const statusResponse = await axios.get(`/api/batch-status/${analysisId}`);
          const status = statusResponse.data;
          
          console.log('狀態回應:', status);

          if (status.status === 'analyzing') {
            const progress = status.progress || 0;
            console.log(`分析中，進度: ${progress}%`);
            setAnalysisProgress(progress);
          } 
          else if (status.status === 'completed') {
            console.log('分析完成！開始獲取結果...');
            clearInterval(pollInterval);
            
            try {
              const resultsUrl = `/api/batch-results/${analysisId}`;
              console.log(`請求結果URL: ${resultsUrl}`);
              
              const resultsResponse = await axios.get(resultsUrl);
              console.log('結果回應狀態:', resultsResponse.status);
              console.log('結果數據:', resultsResponse.data);
              
              // 檢查回應數據
              if (resultsResponse.data && resultsResponse.data.results) {
                console.log(`成功獲取 ${resultsResponse.data.results.length} 筆結果`);
                setAnalysisResults(resultsResponse.data);
                setAnalyzing(false);
                setAnalysisProgress(100);
                
                notification.success({
                  message: '分析完成！',
                  description: `成功分析了 ${resultsResponse.data.statistics?.total_records || '未知數量'} 筆記錄`
                });
              } else {
                console.error('結果數據格式異常:', resultsResponse.data);
                throw new Error('結果數據格式不正確');
              }
            } catch (resultError) {
              console.error('獲取結果失敗:', resultError);
              console.error('錯誤詳情:', resultError.response?.data);
              
              setAnalyzing(false);
              notification.error({
                message: '無法獲取分析結果',
                description: `錯誤: ${resultError.response?.data?.error || resultError.message}`
              });
            }
          }
          else if (status.status === 'error') {
            console.error('分析失敗:', status.error);
            clearInterval(pollInterval);
            setAnalyzing(false);
            notification.error({
              message: '分析失敗',
              description: status.error || '發生未知錯誤'
            });
          }
        } catch (error) {
          console.error('輪詢狀態失敗:', error);
          
          // 如果是404錯誤，停止輪詢
          if (error.response?.status === 404) {
            clearInterval(pollInterval);
            setAnalyzing(false);
            notification.error({
              message: '分析會話不存在',
              description: '分析會話可能已過期。'
            });
          }
          // 其他錯誤繼續輪詢，但記錄錯誤
        }
      }, 2000);

      // 5分鐘超時保護
      setTimeout(() => {
        clearInterval(pollInterval);
        if (analyzing) {
          setAnalyzing(false);
          console.log('分析超時');
          notification.warning({
            message: '分析超時',
            description: '分析時間過長，請手動檢查狀態。'
          });
        }
      }, 300000);

    } catch (error) {
      console.error('啟動分析失敗:', error);
      setAnalyzing(false);
      notification.error({
        message: '啟動分析失敗',
        description: error.response?.data?.error || '無法啟動分析'
      });
    }
  };

  const downloadResults = async () => {
    if (!analysisId) return;

    try {
      const response = await axios.get(`/api/download-results/${analysisId}`);
      const { csv_data, filename } = response.data;
      
      const blob = new Blob([csv_data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      notification.success({
        message: 'Download Started',
        description: 'Results file download has started.'
      });
    } catch (error) {
      notification.error({
        message: 'Download Failed',
        description: 'Failed to download results'
      });
    }
  };

  const getUniqueModels = () => {
    return uploadedData?.all_models || [];
  };

  const getRiskColor = (riskLevel) => {
    const colors = {
      'safe': '#52c41a',
      'low': '#1890ff', 
      'medium': '#faad14',
      'high': '#ff4d4f',
      'critical': '#a8071a'
    };
    return colors[riskLevel] || '#666';
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <div>
      <Title level={2}>📊 Batch Dataset Analysis</Title>
      
      {/* File Upload Section */}
      <Card title="📁 Upload Dataset" style={{ marginBottom: 24 }}>
        <Dragger {...uploadProps} style={{ marginBottom: 16 }}>
          <p className="ant-upload-drag-icon">
            <FileExcelOutlined style={{ fontSize: 48, color: '#1890ff' }} />
          </p>
          <p className="ant-upload-text">Click or drag CSV file to this area to upload</p>
          <p className="ant-upload-hint">
            Upload your jailbreak dataset with columns: Model, Resource, Strategy, Group, Prompt, Response, ToxicityScore, JailbreakSuccess
          </p>
        </Dragger>

        {uploadedData && (
          <Alert
            message={`✅ Dataset loaded successfully! ${uploadedData.total_records} records found.`}
            type="success"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}
      </Card>

      {/* Data Preview */}
      {uploadedData && (
        <Card title="📋 Data Preview" style={{ marginBottom: 24 }}>
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={6}>
              <Statistic title="Total Records" value={uploadedData.total_records} />
            </Col>
            <Col span={6}>
              <Statistic title="Unique Models" value={uploadedData.unique_models} />
            </Col>
            <Col span={6}>
              <Statistic 
                title="Jailbreak Success Rate" 
                value={uploadedData.jailbreak_rate} 
                suffix="%" 
                precision={1}
              />
            </Col>
            <Col span={6}>
              <Statistic title="Available Columns" value={uploadedData.columns?.length} />
            </Col>
          </Row>

          <Table
            dataSource={uploadedData.preview}
            columns={uploadedData.columns?.map(col => ({
              title: col,
              dataIndex: col,  // ✅ 改為 dataIndex
              key: col,
              ellipsis: true,
              width: col === 'Response' || col === 'Prompt' ? 200 : 120
            }))}
            pagination={false}
            scroll={{ x: 1200 }}
            size="small"
          />
        </Card>
      )}

      {/* Analysis Configuration */}
      {uploadedData && (
        <Card title="⚙️ Analysis Configuration" style={{ marginBottom: 24 }}>
          <Row gutter={16}>
            <Col span={12}>
              <div style={{ marginBottom: 16 }}>
                <Text strong>Select Models to Analyze:</Text>
                <Select
                  mode="multiple"
                  placeholder="Choose models"
                  value={selectedModels}
                  onChange={setSelectedModels}
                  style={{ width: '100%', marginTop: 8 }}
                >
                  {getUniqueModels().map(model => (
                    <Option key={model} value={model}>{model}</Option>
                  ))}
                </Select>
              </div>

              <div style={{ marginBottom: 16 }}>
                <Text strong>Sample Size for Analysis:</Text>
                
                {/* 選擇模式：預設 vs 自訂 */}
                <div style={{ marginTop: 8, marginBottom: 8 }}>
                  <Radio.Group 
                    value={sampleSizeMode} 
                    onChange={(e) => setSampleSizeMode(e.target.value)}
                    size="small"
                  >
                    <Radio value="preset">預設選項</Radio>
                    <Radio value="custom">自訂數量</Radio>
                  </Radio.Group>
                </div>

                {/* 預設選項 */}
                {sampleSizeMode === 'preset' && (
                  <Select
                    value={sampleSize}
                    onChange={setSampleSize}
                    style={{ width: '100%' }}
                  >
                    <Option value="50">50 records</Option>
                    <Option value="100">100 records</Option>
                    <Option value="200">200 records</Option>
                    <Option value="500">500 records</Option>
                    <Option value="1000">1000 records</Option>
                    <Option value="2000">2000 records</Option>
                    <Option value="all">All records</Option>
                  </Select>
                )}

                {/* 自訂選項 */}
                {sampleSizeMode === 'custom' && (
                  <div>
                    <InputNumber
                      value={customSampleSize}
                      onChange={setCustomSampleSize}
                      placeholder="輸入記錄數量"
                      min={10}
                      max={uploadedData?.total_records || 10000}
                      style={{ width: '100%' }}
                      formatter={value => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                      parser={value => value.replace(/\$\s?|(,*)/g, '')}
                    />
                    <div style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
                      範圍：10 - {(uploadedData?.total_records || 10000).toLocaleString()} 記錄
                    </div>
                  </div>
                )}
              </div>
            </Col>

            <Col span={12}>
              <div style={{ marginBottom: 16 }}>
                <Checkbox
                  checked={generateAlternatives}
                  onChange={(e) => setGenerateAlternatives(e.target.checked)}
                >
                  <Text strong>Generate Alternative Responses</Text>
                </Checkbox>
                <div style={{ marginTop: 4 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    Generate safe alternative responses for blocked content (slower processing)
                  </Text>
                </div>
              </div>

              <div style={{ marginTop: 32 }}>
                <Button
                  type="primary"
                  size="large"
                  icon={<PlayCircleOutlined />}
                  onClick={startAnalysis}
                  loading={analyzing}
                  disabled={selectedModels.length === 0 || (sampleSizeMode === 'custom' && !customSampleSize)}
                  block
                >
                  🚀 Start Batch Analysis
                </Button>
              </div>
            </Col>
          </Row>

          {analyzing && (
            <div style={{ marginTop: 16 }}>
              <Text strong>🔄 Analysis Progress:</Text>
              <Progress 
                percent={analysisProgress} 
                status={analyzing ? "active" : "success"}
                style={{ marginTop: 8 }}
              />
            </div>
          )}
        </Card>
      )}

      {/* Analysis Results Dashboard */}
      {analysisResults && (
        <Card title="📈 Analysis Results Dashboard" style={{ marginBottom: 24 }}>
          <Tabs defaultActiveKey="overview">
            <TabPane tab="📊 Overview" key="overview">
              <Row gutter={16} style={{ marginBottom: 24 }}>
                <Col span={6}>
                  <Card>
                    <Statistic
                      title="Total Analyzed"
                      value={analysisResults.statistics.total_records}
                    />
                  </Card>
                </Col>
                <Col span={6}>
                  <Card>
                    <Statistic
                      title="Defense Detection Rate"
                      value={analysisResults.statistics.defense_detection_rate}
                      suffix="%"
                      precision={1}
                    />
                  </Card>
                </Col>
                <Col span={6}>
                  <Card>
                    <Statistic
                      title="Avg Toxicity Score"
                      value={analysisResults.statistics.average_toxicity_score}
                      precision={3}
                    />
                  </Card>
                </Col>
                <Col span={6}>
                  <Card>
                    <Statistic
                      title="Alternative Gen. Rate"
                      value={analysisResults.statistics.alternative_generation_rate}
                      suffix="%"
                      precision={1}
                    />
                  </Card>
                </Col>
              </Row>

              {/* Risk Distribution */}
              <Row gutter={16}>
                <Col span={12}>
                  <Card title="Risk Level Distribution">
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={Object.entries(analysisResults.statistics.risk_distribution).map(([level, count]) => ({
                            name: level.toUpperCase(),
                            value: count,
                            color: getRiskColor(level)
                          }))}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {Object.entries(analysisResults.statistics.risk_distribution).map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="Defense vs Original Jailbreak">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart
                        data={[
                          {
                            name: 'Detection Rate',
                            defense: analysisResults.statistics.defense_detection_rate,
                            original: analysisResults.statistics.original_jailbreak_rate || 0
                          }
                        ]}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="defense" fill="#8884d8" name="Defense Detection" />
                        <Bar dataKey="original" fill="#82ca9d" name="Original Jailbreak" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                </Col>
              </Row>
            </TabPane>

            <TabPane tab="🏆 Model Comparison" key="models">
              {analysisResults.statistics.model_comparison && (
                <div>
                  <Alert
                    message="🏆 Model Safety Ranking (Lower jailbreak rate = safer)"
                    type="info"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                  
                  <Table
                    dataSource={Object.entries(analysisResults.statistics.model_comparison).map(([model, stats], index) => ({
                      key: model,
                      rank: index + 1,
                      model,
                      jailbreak_rate: (stats.jailbreak_rate * 100).toFixed(1),
                      sample_count: stats.sample_count,
                      defense_rate: (stats.defense_rate * 100).toFixed(1),
                      avg_toxicity: stats.avg_toxicity.toFixed(3)
                    }))}
                    columns={[
                      {
                        title: 'Rank',
                        dataIndex: 'rank',
                        key: 'rank',
                        render: (rank) => (
                          <Tag color={rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'orange' : 'default'}>
                            #{rank}
                          </Tag>
                        )
                      },
                      {
                        title: 'Model',
                        dataIndex: 'model',
                        key: 'model',
                      },
                      {
                        title: 'Jailbreak Success Rate (%)',
                        dataIndex: 'jailbreak_rate',
                        key: 'jailbreak_rate',
                        render: (rate) => (
                          <Tag color={rate < 20 ? 'green' : rate < 50 ? 'orange' : 'red'}>
                            {rate}%
                          </Tag>
                        )
                      },
                      {
                        title: 'Sample Count',
                        dataIndex: 'sample_count',
                        key: 'sample_count',
                      },
                      {
                        title: 'Defense Detection Rate (%)',
                        dataIndex: 'defense_rate',
                        key: 'defense_rate',
                      },
                      {
                        title: 'Avg Toxicity Score',
                        dataIndex: 'avg_toxicity',
                        key: 'avg_toxicity',
                      }
                    ]}
                    pagination={false}
                  />

                  <div style={{ marginTop: 24 }}>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart
                        data={Object.entries(analysisResults.statistics.model_comparison).map(([model, stats]) => ({
                          model,
                          jailbreak_rate: stats.jailbreak_rate * 100,
                          defense_rate: stats.defense_rate * 100
                        }))}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="model" angle={-45} textAnchor="end" height={100} />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="jailbreak_rate" fill="#ff4d4f" name="Jailbreak Success Rate %" />
                        <Bar dataKey="defense_rate" fill="#52c41a" name="Defense Detection Rate %" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </TabPane>

            <TabPane tab="📋 Detailed Results" key="details">
              <div style={{ marginBottom: 16 }}>
                <Space>
                  <Button 
                    type="primary" 
                    icon={<DownloadOutlined />}
                    onClick={downloadResults}
                  >
                    📊 Download Full Results
                  </Button>
                  <Text>Export complete analysis results with all new defense metrics</Text>
                </Space>
              </div>

              <Table
                dataSource={analysisResults.results?.slice(0, 100)} // Show first 100 for performance
                columns={[
                  {
                    title: 'Model',
                    dataIndex: 'Model',
                    key: 'Model',
                    width: 120,
                  },
                  {
                    title: 'Prompt',
                    dataIndex: 'Prompt',
                    key: 'Prompt',
                    ellipsis: true,
                    width: 200,
                  },
                  {
                    title: 'Response',
                    dataIndex: 'Response',
                    key: 'Response',
                    ellipsis: true,
                    width: 200,
                  },
                  {
                    title: 'Original JB',
                    dataIndex: 'JailbreakSuccess',
                    key: 'JailbreakSuccess',
                    render: (success) => (
                      <Tag color={success ? 'red' : 'green'}>
                        {success ? 'SUCCESS' : 'FAILED'}
                      </Tag>
                    )
                  },
                  {
                    title: 'Defense Risk',
                    dataIndex: 'DefenseRiskLevel',
                    key: 'DefenseRiskLevel',
                    render: (level) => (
                      <Tag color={getRiskColor(level)}>
                        {level?.toUpperCase()}
                      </Tag>
                    )
                  },
                  {
                    title: 'Toxicity',
                    dataIndex: 'DefenseToxicityScore',
                    key: 'DefenseToxicityScore',
                    render: (score) => score?.toFixed(3)
                  },
                  {
                    title: 'Blocked',
                    dataIndex: 'DefenseBlocked',
                    key: 'DefenseBlocked',
                    render: (blocked) => (
                      <Tag color={blocked ? 'red' : 'green'}>
                        {blocked ? 'BLOCKED' : 'PASSED'}
                      </Tag>
                    )
                  }
                ]}
                scroll={{ x: 1200 }}
                pagination={{
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} items (showing first 100)`
                }}
                size="small"
              />
            </TabPane>
          </Tabs>
        </Card>
      )}
    </div>
  );
};

export default BatchAnalysis;
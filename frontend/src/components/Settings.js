import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Form, 
  Input, 
  Button, 
  Slider, 
  Select, 
  Switch, 
  Row, 
  Col,
  Alert,
  notification,
  Typography,
  Divider,
  Space
} from 'antd';
import { SaveOutlined, ReloadOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Option } = Select;
const { Title, Text } = Typography;

const Settings = ({ systemStatus, onUpdate }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [config, setConfig] = useState({});

  const fetchConfig = async () => {
    try {
      const response = await axios.get('/api/config');
      setConfig(response.data);
      form.setFieldsValue(response.data);
    } catch (error) {
      notification.error({
        message: 'Failed to Load Configuration',
        description: 'Could not retrieve system configuration'
      });
    }
  };

  const saveConfig = async (values) => {
    setLoading(true);
    try {
      const response = await axios.post('/api/config', values);
      if (response.data.success) {
        setConfig(values);
        notification.success({
          message: 'Configuration Saved',
          description: 'System configuration updated successfully'
        });
        onUpdate(); // Refresh system status
      } else {
        notification.warning({
          message: 'Configuration Updated with Warnings',
          description: response.data.message
        });
      }
    } catch (error) {
      notification.error({
        message: 'Failed to Save Configuration',
        description: error.response?.data?.error || 'Could not save configuration'
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfig();
  }, []);

  return (
    <div>
      <Title level={2}>⚙️ System Settings</Title>
      
      <Row gutter={24}>
        <Col span={16}>
          <Card title="🔧 Defense Parameters">
            <Form
              form={form}
              layout="vertical"
              onFinish={saveConfig}
              initialValues={config}
            >
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="Toxicity Detection Threshold"
                    name="toxicity_threshold"
                    tooltip="Controls sensitivity of toxicity detection (0.0-1.0)"
                  >
                    <Slider
                      min={0}
                      max={1}
                      step={0.01}
                      marks={{
                        0: '0.0',
                        0.3: '0.3',
                        0.5: '0.5',
                        0.7: '0.7',
                        1: '1.0'
                      }}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="Harmfulness Detection Threshold"
                    name="harmfulness_threshold"
                    tooltip="Controls sensitivity of harmful content detection (0.0-1.0)"
                  >
                    <Slider
                      min={0}
                      max={1}
                      step={0.01}
                      marks={{
                        0: '0.0',
                        0.2: '0.2',
                        0.5: '0.5',
                        0.8: '0.8',
                        1: '1.0'
                      }}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="Alternative Response Trigger Level"
                    name="replacement_threshold"
                    tooltip="Determines which risk level triggers Ollama alternative generation"
                  >
                    <Select>
                      <Option value="low">Low risk and above</Option>
                      <Option value="medium">Medium risk and above</Option>
                      <Option value="high">High risk and above</Option>
                      <Option value="critical">Critical risk only</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="Enable Ollama Alternative Response"
                    name="enable_ollama_replacement"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                  <Form.Item
                    label="Strict Mode"
                    name="strict_mode"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
              </Row>

              <Divider />

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="Toxicity Model Path"
                    name="model_path"
                    tooltip="Path to your toxicity detection model"
                  >
                    <Input placeholder="C:/path/to/your/model" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="Ollama Service URL"
                    name="ollama_url"
                    tooltip="Local Ollama service address"
                  >
                    <Input placeholder="http://localhost:11434" />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                label="Default Ollama Model"
                name="ollama_model"
                tooltip="Default model for alternative response generation"
              >
                <Select>
                  {systemStatus.available_models?.map(model => (
                    <Option key={model} value={model}>{model}</Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item>
                <Space>
                  <Button 
                    type="primary" 
                    htmlType="submit" 
                    icon={<SaveOutlined />}
                    loading={loading}
                  >
                    Save Configuration
                  </Button>
                  <Button 
                    icon={<ReloadOutlined />}
                    onClick={fetchConfig}
                  >
                    Reset to Current
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        <Col span={8}>
          <Card title="📊 System Status">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="Model Status"
                description={systemStatus.model_exists ? 
                  "✅ Toxicity detection model found" : 
                  "❌ Toxicity detection model not found"
                }
                type={systemStatus.model_exists ? "success" : "error"}
                showIcon
              />
              
              <Alert
                message="Ollama Service"
                description={systemStatus.ollama_connected ? 
                  "✅ Ollama service connected" : 
                  "❌ Cannot connect to Ollama service"
                }
                type={systemStatus.ollama_connected ? "success" : "error"}
                showIcon
              />
              
              <Alert
                message="Defense System"
                description={systemStatus.defense_system_ready ? 
                  "✅ Defense system active" : 
                  "❌ Defense system not ready"
                }
                type={systemStatus.defense_system_ready ? "success" : "error"}
                showIcon
              />

              {systemStatus.available_models?.length > 0 && (
                <Card size="small" title="Available Models">
                  {systemStatus.available_models.map(model => (
                    <div key={model} style={{ marginBottom: 4 }}>
                      <Text code>{model}</Text>
                    </div>
                  ))}
                </Card>
              )}
            </Space>
          </Card>

          <Card title="💡 Configuration Tips" style={{ marginTop: 16 }}>
            <ul style={{ paddingLeft: 16 }}>
              <li>Lower thresholds = more sensitive detection</li>
              <li>Higher thresholds = fewer false positives</li>
              <li>Strict mode enables enhanced security checking</li>
              <li>Alternative responses provide constructive feedback</li>
              <li>Test different settings with the real-time testing feature</li>
            </ul>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Settings;
import React, { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Form,
  Select,
  Switch,
  Space,
  Typography,
  Alert,
  Divider,
  Row,
  Col,
  Statistic,
  Tag,
  List,
  Modal,
  message,
  Progress,
} from 'antd'
import {
  ThunderboltOutlined,
  DeleteOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  RocketOutlined,
} from '@ant-design/icons'
import { useAppStore } from '../store/appStore'
import apiService from '../services/api'
import type { ModelInfo, SystemInfo } from '../types'

const { Title, Text } = Typography
const { Option } = Select

const SettingsPage: React.FC = () => {
  const [form] = Form.useForm()
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const { currentModel, setCurrentModel, setSystemStatus } = useAppStore()

  // 加载模型列表
  const loadModels = async () => {
    try {
      setLoading(true)
      const response = await apiService.getModels()
      if (response.success && response.data) {
        setModels(response.data)
      }
    } catch (error) {
      console.error('加载模型列表失败:', error)
      message.error('加载模型列表失败')
    } finally {
      setLoading(false)
    }
  }

  // 加载系统状态
  const loadSystemStatus = async () => {
    try {
      const response = await apiService.getSystemStatus()
      if (response.success && response.data) {
        setSystemInfo(response.data)
        setSystemStatus(response.data)
        if (response.data.current_model) {
          setCurrentModel(response.data.current_model)
        }
      }
    } catch (error) {
      console.error('加载系统状态失败:', error)
    }
  }

  // 初始化数据
  useEffect(() => {
    loadModels()
    loadSystemStatus()
  }, [])

  // 加载模型
  const handleLoadModel = async (modelName: string) => {
    try {
      setLoading(true)
      const response = await apiService.loadModel(modelName)
      if (response.success) {
        setCurrentModel(modelName)
        message.success(`模型 ${modelName} 加载成功`)
        await loadSystemStatus() // 重新加载系统状态
      }
    } catch (error) {
      console.error('加载模型失败:', error)
      message.error('加载模型失败')
    } finally {
      setLoading(false)
    }
  }

  // 删除模型
  const handleDeleteModel = (modelName: string) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除模型 "${modelName}" 吗？此操作不可恢复。`,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          const response = await apiService.deleteModel(modelName)
          if (response.success) {
            message.success('模型删除成功')
            await loadModels() // 重新加载模型列表
            if (currentModel === modelName) {
              setCurrentModel(null)
            }
          }
        } catch (error) {
          console.error('删除模型失败:', error)
          message.error('删除模型失败')
        }
      },
    })
  }

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // 格式化内存使用率
  const getMemoryUsage = () => {
    if (!systemInfo?.gpu_memory) return 0
    const { used, total } = systemInfo.gpu_memory
    return total > 0 ? Math.round((used / total) * 100) : 0
  }

  return (
    <div className="p-6 space-y-6">
      <div className="mb-6">
        <Title level={2} className="!mb-2">
          系统设置
        </Title>
        <Text type="secondary">
          管理模型、查看系统状态和配置参数
        </Text>
      </div>

      <Row gutter={24}>
        {/* 左侧 - 系统状态 */}
        <Col xs={24} lg={12}>
          <Card title="系统状态" className="h-full">
            <Space direction="vertical" className="w-full" size="large">
              {/* 基本信息 */}
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="当前模型"
                    value={currentModel || '未加载'}
                    prefix={currentModel ? <CheckCircleOutlined style={{ color: '#52c41a' }} /> : <ExclamationCircleOutlined style={{ color: '#faad14' }} />}
                    valueStyle={{ 
                      color: currentModel ? '#52c41a' : '#faad14',
                      fontSize: '16px'
                    }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="可用模型"
                    value={models.length}
                    suffix="个"
                    prefix={<RocketOutlined />}
                    valueStyle={{ color: '#1890ff', fontSize: '16px' }}
                  />
                </Col>
              </Row>

              {/* GPU 状态 */}
              <div>
                <Text strong className="block mb-2">GPU 状态</Text>
                <Space direction="vertical" className="w-full">
                  <div className="flex justify-between">
                    <Text>GPU 可用性:</Text>
                    <Tag color={systemInfo?.gpu_available ? 'success' : 'error'}>
                      {systemInfo?.gpu_available ? '可用' : '不可用'}
                    </Tag>
                  </div>
                  
                  {systemInfo?.gpu_available && systemInfo.gpu_memory && (
                    <div>
                      <div className="flex justify-between mb-2">
                        <Text>GPU 内存:</Text>
                        <Text>{getMemoryUsage()}%</Text>
                      </div>
                      <Progress
                        percent={getMemoryUsage()}
                        strokeColor={{
                          '0%': '#52c41a',
                          '70%': '#faad14',
                          '90%': '#ff4d4f',
                        }}
                        size="small"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>{formatFileSize(systemInfo.gpu_memory.used)}</span>
                        <span>{formatFileSize(systemInfo.gpu_memory.total)}</span>
                      </div>
                    </div>
                  )}
                </Space>
              </div>

              {/* 系统资源 */}
              {systemInfo && (
                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic
                      title="CPU 使用率"
                      value={systemInfo.cpu_usage}
                      precision={1}
                      suffix="%"
                      valueStyle={{ 
                        color: systemInfo.cpu_usage > 80 ? '#ff4d4f' : '#52c41a',
                        fontSize: '16px'
                      }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="内存使用率"
                      value={systemInfo.memory_usage}
                      precision={1}
                      suffix="%"
                      valueStyle={{ 
                        color: systemInfo.memory_usage > 80 ? '#ff4d4f' : '#52c41a',
                        fontSize: '16px'
                      }}
                    />
                  </Col>
                </Row>
              )}

              <Button
                icon={<ReloadOutlined />}
                onClick={loadSystemStatus}
                type="default"
                className="w-full"
              >
                刷新状态
              </Button>
            </Space>
          </Card>
        </Col>

        {/* 右侧 - 模型管理 */}
        <Col xs={24} lg={12}>
          <Card 
            title="模型管理" 
            extra={
              <Button 
                icon={<ReloadOutlined />} 
                onClick={loadModels}
                loading={loading}
                size="small"
              >
                刷新
              </Button>
            }
            className="h-full"
          >
            <Space direction="vertical" className="w-full" size="large">
              {models.length === 0 ? (
                <div className="text-center py-8">
                  <RocketOutlined className="text-4xl text-gray-300 mb-2" />
                  <Text type="secondary">没有找到可用的模型</Text>
                </div>
              ) : (
                <List
                  dataSource={models}
                  renderItem={(model) => (
                    <List.Item
                      actions={[
                        <Button
                          key="load"
                          type={currentModel === model.name ? 'default' : 'primary'}
                          size="small"
                          icon={currentModel === model.name ? <CheckCircleOutlined /> : <ThunderboltOutlined />}
                          onClick={() => handleLoadModel(model.name)}
                          disabled={currentModel === model.name || loading}
                        >
                          {currentModel === model.name ? '已加载' : '加载'}
                        </Button>,
                        <Button
                          key="delete"
                          danger
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={() => handleDeleteModel(model.name)}
                          disabled={currentModel === model.name}
                        >
                          删除
                        </Button>,
                      ]}
                    >
                      <List.Item.Meta
                        title={
                          <Space>
                            <Text strong>{model.name}</Text>
                            {currentModel === model.name && (
                              <Tag color="success">当前</Tag>
                            )}
                          </Space>
                        }
                        description={
                          <Space direction="vertical" size="small">
                            <Text type="secondary" className="text-sm">
                              大小: {formatFileSize(model.size)}
                            </Text>
                            {model.accuracy && (
                              <Text type="secondary" className="text-sm">
                                准确率: {model.accuracy.toFixed(2)}%
                              </Text>
                            )}
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />
              )}

              <Divider />

              {/* 模型配置 */}
              <div>
                <Text strong className="block mb-3">模型配置</Text>
                <Form form={form} layout="vertical">
                  <Form.Item label="默认模型" name="default_model">
                    <Select
                      placeholder="选择默认加载的模型"
                      value={currentModel}
                      onChange={handleLoadModel}
                    >
                      {models.map(model => (
                        <Option key={model.name} value={model.name}>
                          {model.name}
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>

                  <Form.Item label="自动加载" name="auto_load">
                    <Switch 
                      defaultChecked={true}
                      checkedChildren="开启"
                      unCheckedChildren="关闭"
                    />
                  </Form.Item>
                </Form>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 警告信息 */}
      {!systemInfo?.gpu_available && (
        <Alert
          message="GPU 不可用"
          description="当前环境未检测到可用的GPU，模型加载和训练将使用CPU，性能可能受到影响。"
          type="warning"
          showIcon
        />
      )}

      {models.length === 0 && (
        <Alert
          message="没有可用的模型"
          description="请先在训练页面训练一个模型，或将预训练模型文件放入 models 文件夹。"
          type="info"
          showIcon
        />
      )}
    </div>
  )
}

export default SettingsPage

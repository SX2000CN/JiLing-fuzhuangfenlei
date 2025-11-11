import React, { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Form,
  Input,
  InputNumber,
  Select,
  Progress,
  Space,
  Typography,
  Alert,
  Divider,
  Row,
  Col,
  Statistic,
  Timeline,
  Tag,
  Spin,
} from 'antd'
import {
  FolderOpenOutlined,
  HistoryOutlined,
  RocketOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  StopOutlined,
} from '@ant-design/icons'
import { useAppStore } from '../store/appStore'
import apiService from '../services/api'
import type { TrainingConfig, TrainingStatus } from '../types'

const { Title, Text } = Typography
const { Option } = Select

const TrainingPage: React.FC = () => {
  const [form] = Form.useForm()
  const [isTraining, setIsTraining] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const { systemStatus } = useAppStore()

  // 轮询训练状态
  useEffect(() => {
    let interval: number | null = null

    if (isTraining) {
        interval = setInterval(async () => {
        try {
          const response = await apiService.getTrainingStatus()
          if (response.success && response.data) {
            setTrainingStatus(response.data)
            if (response.data.logs) {
              setLogs(response.data.logs)
            }
            
            // 检查训练是否完成
            if (!response.data.isTraining) {
              setIsTraining(false)
            }
          }
        } catch (error) {
          console.error('获取训练状态失败:', error)
        }
      }, 2000) // 每2秒更新一次状态
    }

    return () => {
      if (interval) {
        clearInterval(interval)
      }
    }
  }, [isTraining])

  // 选择数据路径
  const handleSelectDataPath = async () => {
    try {
      const response = await apiService.selectFolder()
      if (response.success && response.data?.path) {
        form.setFieldsValue({ data_path: response.data.path })
      }
    } catch (error) {
      console.error('选择文件夹失败:', error)
    }
  }

  // 开始训练
  const handleStartTraining = async (values: TrainingConfig) => {
    try {
      setIsTraining(true)
      setLogs([])
      
      const response = await apiService.startTraining(values)
      if (response.success) {
        console.log('训练启动成功')
      }
    } catch (error) {
      setIsTraining(false)
      console.error('启动训练失败:', error)
    }
  }

  // 停止训练
  const handleStopTraining = async () => {
    try {
      const response = await apiService.stopTraining()
      if (response.success) {
        setIsTraining(false)
        console.log('训练已停止')
      }
    } catch (error) {
      console.error('停止训练失败:', error)
    }
  }

  // 计算训练进度
  const getTrainingProgress = () => {
    if (!trainingStatus) return 0
    return trainingStatus.totalEpochs > 0 
      ? Math.round((trainingStatus.currentEpoch / trainingStatus.totalEpochs) * 100)
      : 0
  }

  // 格式化训练时间
  const formatTrainingTime = (startTime: number) => {
    if (!startTime) return '00:00:00'
    const elapsed = Date.now() / 1000 - startTime
    const hours = Math.floor(elapsed / 3600)
    const minutes = Math.floor((elapsed % 3600) / 60)
    const seconds = Math.floor(elapsed % 60)
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
  }

  return (
    <div className="p-6 space-y-6">
      <div className="mb-6">
        <Title level={2} className="!mb-2">
          模型训练
        </Title>
        <Text type="secondary">
          配置并启动模型训练任务
        </Text>
      </div>

      <Row gutter={24}>
        {/* 左侧 - 训练配置 */}
        <Col xs={24} lg={12}>
          <Card title="训练配置" className="h-full">
            <Form
              form={form}
              layout="vertical"
              onFinish={handleStartTraining}
              initialValues={{
                epochs: 10,
                learning_rate: 0.001,
                batch_size: 32,
                validation_split: 0.2,
              }}
            >
              <Form.Item
                label="数据路径"
                name="data_path"
                rules={[{ required: true, message: '请选择训练数据路径' }]}
              >
                <Input
                  placeholder="选择包含训练数据的文件夹"
                  readOnly
                  suffix={
                    <Button
                      type="text"
                      icon={<FolderOpenOutlined />}
                      onClick={handleSelectDataPath}
                      size="small"
                    >
                      浏览
                    </Button>
                  }
                />
              </Form.Item>

              <Form.Item
                label="模型名称"
                name="model_name"
                rules={[{ required: true, message: '请输入模型名称' }]}
              >
                <Input placeholder="新模型的名称" />
              </Form.Item>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="训练轮数"
                    name="epochs"
                    rules={[{ required: true, message: '请输入训练轮数' }]}
                  >
                    <InputNumber
                      min={1}
                      max={1000}
                      className="w-full"
                      placeholder="10"
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="学习率"
                    name="learning_rate"
                    rules={[{ required: true, message: '请输入学习率' }]}
                  >
                    <InputNumber
                      min={0.0001}
                      max={1}
                      step={0.0001}
                      precision={4}
                      className="w-full"
                      placeholder="0.001"
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="批次大小"
                    name="batch_size"
                    rules={[{ required: true, message: '请输入批次大小' }]}
                  >
                    <Select placeholder="选择批次大小">
                      <Option value={16}>16</Option>
                      <Option value={32}>32</Option>
                      <Option value={64}>64</Option>
                      <Option value={128}>128</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="验证集比例"
                    name="validation_split"
                    rules={[{ required: true, message: '请输入验证集比例' }]}
                  >
                    <InputNumber
                      min={0.1}
                      max={0.5}
                      step={0.05}
                      precision={2}
                      className="w-full"
                      placeholder="0.2"
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Divider />

              <Space className="w-full justify-center">
                <Button
                  type="primary"
                  size="large"
                  icon={<RocketOutlined />}
                  htmlType="submit"
                  loading={isTraining}
                  disabled={isTraining}
                  className="min-w-32"
                >
                  {isTraining ? '训练中...' : '开始训练'}
                </Button>
                {isTraining && (
                  <Button
                    danger
                    size="large"
                    icon={<StopOutlined />}
                    onClick={handleStopTraining}
                    className="min-w-24"
                  >
                    停止
                  </Button>
                )}
              </Space>
            </Form>
          </Card>
        </Col>

        {/* 右侧 - 训练状态和日志 */}
        <Col xs={24} lg={12}>
          <Card title="训练状态" className="h-full">
            {!isTraining && !trainingStatus && (
              <div className="text-center py-12">
                <RocketOutlined className="text-6xl text-gray-300 mb-4" />
                <Text type="secondary" className="text-lg">
                  等待开始训练
                </Text>
              </div>
            )}

            {isTraining && (
              <Spin spinning={!trainingStatus} tip="正在启动训练...">
                <Space direction="vertical" className="w-full" size="large">
                  {/* 训练进度 */}
                  {trainingStatus && (
                    <>
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <Text strong>训练进度</Text>
                          <Tag color="processing" icon={<ClockCircleOutlined />}>
                            进行中
                          </Tag>
                        </div>
                        <Progress
                          percent={getTrainingProgress()}
                          strokeColor={{
                            '0%': '#108ee9',
                            '100%': '#87d068',
                          }}
                          format={(percent) => `${trainingStatus.currentEpoch}/${trainingStatus.totalEpochs} (${percent}%)`}
                        />
                      </div>

                      {/* 训练统计 */}
                      <Row gutter={16}>
                        <Col span={8}>
                          <Statistic
                            title="当前轮次"
                            value={trainingStatus.currentEpoch}
                            suffix={`/ ${trainingStatus.totalEpochs}`}
                            valueStyle={{ fontSize: '18px' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Statistic
                            title="当前损失"
                            value={trainingStatus.currentLoss}
                            precision={4}
                            valueStyle={{ fontSize: '18px', color: '#faad14' }}
                          />
                        </Col>
                        <Col span={8}>
                          <Statistic
                            title="最佳准确率"
                            value={trainingStatus.bestAcc}
                            precision={2}
                            suffix="%"
                            valueStyle={{ fontSize: '18px', color: '#52c41a' }}
                          />
                        </Col>
                      </Row>

                      {/* 训练时间 */}
                      {trainingStatus.startTime && (
                        <div className="text-center p-2 bg-gray-50 rounded">
                          <Text type="secondary">
                            训练时间: {formatTrainingTime(trainingStatus.startTime)}
                          </Text>
                        </div>
                      )}
                    </>
                  )}
                </Space>
              </Spin>
            )}

            {/* 训练日志 */}
            {logs.length > 0 && (
              <div className="mt-6">
                <Divider orientation="left">
                  <Space>
                    <HistoryOutlined />
                    <Text strong>训练日志</Text>
                  </Space>
                </Divider>
                
                <div className="max-h-64 overflow-y-auto bg-gray-50 p-3 rounded border">
                  <Timeline
                    mode="left"
                    items={logs.map((log) => ({
                      color: log.includes('完成') ? 'green' : 
                             log.includes('失败') || log.includes('错误') ? 'red' : 'blue',
                      children: (
                        <Text
                          className="text-sm"
                          type={log.includes('失败') || log.includes('错误') ? 'danger' : 'secondary'}
                        >
                          {log}
                        </Text>
                      ),
                    }))}
                  />
                </div>
              </div>
            )}

            {/* 完成状态 */}
            {!isTraining && trainingStatus && (
              <Alert
                message="训练完成"
                description="模型训练已完成，可以在模型管理中查看和使用新模型"
                type="success"
                icon={<CheckCircleOutlined />}
                showIcon
                className="mt-4"
              />
            )}
          </Card>
        </Col>
      </Row>

      {/* GPU 状态提示 */}
      {!systemStatus?.gpu_available && (
        <Alert
          message="GPU 不可用"
          description="当前环境未检测到可用的GPU，训练将使用CPU，速度可能较慢"
          type="warning"
          showIcon
          className="mt-4"
        />
      )}
    </div>
  )
}

export default TrainingPage

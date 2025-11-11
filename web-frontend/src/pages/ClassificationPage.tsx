import React, { useState } from 'react'
import {
  Card,
  Button,
  Space,
  Image,
  Typography,
  Progress,
  Alert,
  Upload,
  message,
  Row,
  Col,
  Statistic,
  Tag,
  Divider,
} from 'antd'
import {
  CameraOutlined,
  FolderOpenOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import type { UploadFile } from 'antd'
import { useAppStore } from '../store/appStore'
import apiService from '../services/api'

const { Title, Text } = Typography
const { Dragger } = Upload

const ClassificationPage: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [isClassifying, setIsClassifying] = useState(false)
  const [classificationResult, setClassificationResult] = useState<any>(null)
  const [uploadFiles, setUploadFiles] = useState<UploadFile[]>([])

  const { systemStatus, currentModel } = useAppStore()

  // 处理文件选择
  const handleFileSelect = async () => {
    try {
      const response = await apiService.selectFile()
      if (response.success && response.data?.path) {
        setSelectedImage(response.data.path)
        setClassificationResult(null)
        message.success('图片加载成功')
      }
    } catch (error) {
      message.error('选择文件失败')
      console.error('File select error:', error)
    }
  }

  // 处理文件上传
  const handleUpload = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string)
      setClassificationResult(null)
    }
    reader.readAsDataURL(file)
    return false // 阻止自动上传
  }

  // 执行分类
  const handleClassify = async () => {
    if (!selectedImage) {
      message.warning('请先选择图片')
      return
    }

    if (!currentModel) {
      message.warning('请先加载模型')
      return
    }

    setIsClassifying(true)
    try {
      const response = await apiService.classifyImage({
        image_path: selectedImage,
        model_name: currentModel,
      })

      if (response.success && response.data) {
        setClassificationResult(response.data)
        message.success('分类完成')
      } else {
        message.error(response.message || '分类失败')
      }
    } catch (error) {
      message.error('分类过程中发生错误')
      console.error('Classification error:', error)
    } finally {
      setIsClassifying(false)
    }
  }

  // 清空结果
  const handleReset = () => {
    setSelectedImage(null)
    setClassificationResult(null)
    setUploadFiles([])
  }

  return (
    <div className="p-6 space-y-6">
      <div className="mb-6">
        <Title level={2} className="!mb-2">
          图像分类
        </Title>
        <Text type="secondary">
          上传或选择图片进行服装分类识别
        </Text>
      </div>

      {/* 系统状态检查 */}
      {!systemStatus?.current_model && (
        <Alert
          message="请先加载模型"
          description="在设置页面选择并加载一个模型后再进行分类"
          type="warning"
          showIcon
          className="mb-4"
        />
      )}

      <Row gutter={24}>
        {/* 左侧 - 图片上传和预览 */}
        <Col xs={24} lg={12}>
          <Card title="图片选择" className="h-full">
            <Space direction="vertical" className="w-full" size="large">
              {/* 上传区域 */}
              <Dragger
                accept="image/*"
                beforeUpload={handleUpload}
                fileList={uploadFiles}
                onChange={({ fileList }) => setUploadFiles(fileList)}
                className="!bg-gray-50"
              >
                <p className="ant-upload-drag-icon">
                  <CameraOutlined className="text-4xl text-blue-500" />
                </p>
                <p className="ant-upload-text">点击或拖拽图片到此区域上传</p>
                <p className="ant-upload-hint">支持 JPG, PNG, BMP 等格式</p>
              </Dragger>

              {/* 操作按钮 */}
              <Space wrap>
                <Button
                  icon={<FolderOpenOutlined />}
                  onClick={handleFileSelect}
                  type="dashed"
                >
                  浏览文件
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={handleReset}
                  type="default"
                >
                  清空
                </Button>
              </Space>

              {/* 图片预览 */}
              {selectedImage && (
                <div className="border rounded-lg p-4 bg-white">
                  <Text strong className="block mb-2">图片预览:</Text>
                  <Image
                    src={selectedImage}
                    alt="Selected"
                    className="max-w-full rounded"
                    style={{ maxHeight: '300px' }}
                  />
                </div>
              )}
            </Space>
          </Card>
        </Col>

        {/* 右侧 - 分类操作和结果 */}
        <Col xs={24} lg={12}>
          <Card title="分类结果" className="h-full">
            <Space direction="vertical" className="w-full" size="large">
              {/* 分类按钮 */}
              <Button
                type="primary"
                size="large"
                icon={<PlayCircleOutlined />}
                onClick={handleClassify}
                loading={isClassifying}
                disabled={!selectedImage || !currentModel}
                className="w-full h-12"
              >
                {isClassifying ? '分类中...' : '开始分类'}
              </Button>

              {/* 分类进度 */}
              {isClassifying && (
                <div>
                  <Text>正在分析图片...</Text>
                  <Progress percent={100} status="active" showInfo={false} />
                </div>
              )}

              {/* 分类结果显示 */}
              {classificationResult && (
                <div className="space-y-4">
                  <Divider orientation="left">分类结果</Divider>
                  
                  {/* 主要结果 */}
                  <Row gutter={16}>
                    <Col span={12}>
                      <Statistic
                        title="预测类别"
                        value={classificationResult.predicted_class}
                        valueStyle={{ color: '#1890ff', fontSize: '20px' }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="置信度"
                        value={classificationResult.confidence}
                        precision={2}
                        suffix="%"
                        valueStyle={{ 
                          color: classificationResult.confidence > 80 ? '#52c41a' : '#faad14',
                          fontSize: '20px'
                        }}
                      />
                    </Col>
                  </Row>

                  {/* 详细概率 */}
                  {classificationResult.probabilities && (
                    <div>
                      <Text strong>各类别概率:</Text>
                      <div className="mt-2 space-y-2">
                        {Object.entries(classificationResult.probabilities)
                          .sort(([,a], [,b]) => (b as number) - (a as number))
                          .slice(0, 5)
                          .map(([className, probability]) => {
                            const maxProb = Math.max(...Object.values(classificationResult.probabilities).map((p: any) => p as number))
                            return (
                              <div key={className} className="flex justify-between items-center">
                                <Tag color={probability === maxProb ? 'blue' : 'default'}>
                                  {className}
                                </Tag>
                                <div className="flex-1 mx-3">
                                  <Progress
                                    percent={Number(probability)}
                                    size="small"
                                    showInfo={false}
                                  />
                                </div>
                                <Text className="text-sm w-12 text-right">
                                  {Number(probability).toFixed(1)}%
                                </Text>
                              </div>
                            )
                          })}
                      </div>
                    </div>
                  )}

                  {/* 处理时间 */}
                  {classificationResult.processing_time && (
                    <div className="text-center p-2 bg-gray-50 rounded">
                      <Text type="secondary">
                        处理时间: {classificationResult.processing_time.toFixed(2)} 秒
                      </Text>
                    </div>
                  )}
                </div>
              )}
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default ClassificationPage

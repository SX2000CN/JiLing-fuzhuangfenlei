import React, { useState } from 'react'
import {
  Card,
  Button,
  Space,
  Typography,
  Table,
  Tag,
  Progress,
  Row,
  Col,
  Statistic,
  Empty,
  Image,
  Tooltip,
} from 'antd'
import {
  BarChartOutlined,
  FileImageOutlined,
  FolderOpenOutlined,
  DownloadOutlined,
  EyeOutlined,
} from '@ant-design/icons'
import { useAppStore } from '../store/appStore'
import apiService from '../services/api'
import type { BatchClassificationResult } from '../types'

const { Title, Text } = Typography

const ResultsPage: React.FC = () => {
  const [results, setResults] = useState<BatchClassificationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const { currentModel } = useAppStore()

  // 批量分类文件夹
  const handleBatchClassify = async () => {
    try {
      const folderResponse = await apiService.selectFolder()
      if (!folderResponse.success || !folderResponse.data?.path) return

      setLoading(true)
      const response = await apiService.batchClassifyImages({
        folder_path: folderResponse.data.path,
        model_name: currentModel || undefined,
      })

      if (response.success && response.data) {
        setResults(response.data)
      }
    } catch (error) {
      console.error('批量分类失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 表格列定义
  const columns = [
    {
      title: '图片',
      dataIndex: 'file_name',
      key: 'file_name',
      width: 120,
      render: (fileName: string) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => setSelectedImage(fileName)}
          size="small"
        >
          查看
        </Button>
      ),
    },
    {
      title: '文件名',
      dataIndex: 'file_name',
      key: 'file_name_text',
      ellipsis: {
        showTitle: false,
      },
      render: (fileName: string) => (
        <Tooltip title={fileName}>
          <Text className="text-sm">{fileName}</Text>
        </Tooltip>
      ),
    },
    {
      title: '预测类别',
      dataIndex: 'category',
      key: 'category',
      width: 120,
      render: (category: string) => (
        <Tag color="blue">{category}</Tag>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence: number) => (
        <Text strong style={{ 
          color: confidence > 0.8 ? '#52c41a' : confidence > 0.6 ? '#faad14' : '#ff4d4f' 
        }}>
          {(confidence * 100).toFixed(1)}%
        </Text>
      ),
      sorter: (a: any, b: any) => a.confidence - b.confidence,
    },
    {
      title: '详细预测',
      dataIndex: 'predictions',
      key: 'predictions',
      render: (predictions: Array<{ category: string; confidence: number }>) => (
        <div className="space-y-1">
          {predictions.slice(0, 3).map((pred, index) => (
            <div key={index} className="flex items-center space-x-2">
              <Text className="text-xs w-16">{pred.category}</Text>
              <Progress
                percent={pred.confidence * 100}
                size="small"
                showInfo={false}
                strokeColor={index === 0 ? '#1890ff' : '#d9d9d9'}
              />
              <Text className="text-xs w-10">
                {(pred.confidence * 100).toFixed(0)}%
              </Text>
            </div>
          ))}
        </div>
      ),
    },
  ]

  // 计算统计信息
  const getStatistics = () => {
    if (!results) return null

    const categoryCount: Record<string, number> = {}
    let highConfidenceCount = 0
    let lowConfidenceCount = 0

    results.results.forEach(result => {
      categoryCount[result.category] = (categoryCount[result.category] || 0) + 1
      if (result.confidence > 0.8) {
        highConfidenceCount++
      } else if (result.confidence < 0.6) {
        lowConfidenceCount++
      }
    })

    const sortedCategories = Object.entries(categoryCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)

    return {
      categoryCount: sortedCategories,
      highConfidenceCount,
      lowConfidenceCount,
      averageConfidence: results.results.reduce((sum, r) => sum + r.confidence, 0) / results.results.length
    }
  }

  const statistics = getStatistics()

  return (
    <div className="p-6 space-y-6">
      <div className="mb-6">
        <Title level={2} className="!mb-2">
          结果查看
        </Title>
        <Text type="secondary">
          查看批量分类结果和统计信息
        </Text>
      </div>

      {/* 操作区域 */}
      <Card>
        <Space size="large">
          <Button
            type="primary"
            icon={<FolderOpenOutlined />}
            onClick={handleBatchClassify}
            loading={loading}
            disabled={!currentModel}
            size="large"
          >
            选择文件夹进行批量分类
          </Button>
          
          {results && (
            <Button
              icon={<DownloadOutlined />}
              onClick={() => {
                // 导出结果逻辑
                const dataStr = JSON.stringify(results, null, 2)
                const dataBlob = new Blob([dataStr], { type: 'application/json' })
                const url = URL.createObjectURL(dataBlob)
                const link = document.createElement('a')
                link.href = url
                link.download = 'classification_results.json'
                link.click()
                URL.revokeObjectURL(url)
              }}
            >
              导出结果
            </Button>
          )}
        </Space>

        {!currentModel && (
          <Text type="warning" className="block mt-2">
            请先在设置页面加载一个模型
          </Text>
        )}
      </Card>

      {/* 结果展示 */}
      {!results ? (
        <Card>
          <Empty
            image={<BarChartOutlined className="text-6xl text-gray-300" />}
            description="暂无分类结果"
          />
        </Card>
      ) : (
        <>
          {/* 统计信息 */}
          {statistics && (
            <Row gutter={24}>
              <Col xs={24} lg={18}>
                <Card title="统计信息">
                  <Row gutter={16}>
                    <Col span={6}>
                      <Statistic
                        title="总计"
                        value={results.total}
                        suffix="张图片"
                        prefix={<FileImageOutlined />}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="已处理"
                        value={results.processed}
                        suffix="张"
                        valueStyle={{ color: '#52c41a' }}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="平均置信度"
                        value={statistics.averageConfidence * 100}
                        precision={1}
                        suffix="%"
                        valueStyle={{ 
                          color: statistics.averageConfidence > 0.8 ? '#52c41a' : 
                                statistics.averageConfidence > 0.6 ? '#faad14' : '#ff4d4f' 
                        }}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="高置信度"
                        value={statistics.highConfidenceCount}
                        suffix={`/ ${results.total}`}
                        valueStyle={{ color: '#52c41a' }}
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
              
              <Col xs={24} lg={6}>
                <Card title="类别分布" className="h-full">
                  <div className="space-y-2">
                    {statistics.categoryCount.map(([category, count]) => (
                      <div key={category} className="flex justify-between items-center">
                        <Tag>{category}</Tag>
                        <Text strong>{count}</Text>
                      </div>
                    ))}
                  </div>
                </Card>
              </Col>
            </Row>
          )}

          {/* 结果表格 */}
          <Card title="详细结果">
            <Table
              dataSource={results.results}
              columns={columns}
              rowKey="file_name"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => 
                  `第 ${range[0]}-${range[1]} 条，共 ${total} 条结果`,
              }}
              scroll={{ x: 800 }}
              size="small"
            />
          </Card>
        </>
      )}

      {/* 图片预览模态框 */}
      {selectedImage && (
        <Image
          style={{ display: 'none' }}
          src={selectedImage}
          preview={{
            visible: !!selectedImage,
            onVisibleChange: (visible) => {
              if (!visible) setSelectedImage(null)
            },
          }}
        />
      )}
    </div>
  )
}

export default ResultsPage

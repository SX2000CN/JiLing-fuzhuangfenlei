import React from 'react'
import { Layout, Space, Badge, Typography, Card } from 'antd'
import { WifiOutlined, CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons'
import { useAppStore } from '../store/appStore'

const { Header } = Layout
const { Text } = Typography

interface HeaderProps {
  sidebarCollapsed: boolean
}

const AppHeader: React.FC<HeaderProps> = ({ sidebarCollapsed }) => {
  const { systemStatus, isConnected } = useAppStore()

  return (
    <Header
      style={{
        background: '#fff',
        padding: '0 24px',
        boxShadow: '0 2px 8px 0 rgba(29, 35, 41, 0.05)',
        borderBottom: '1px solid #f0f0f0',
        position: 'fixed',
        top: 0,
        right: 0,
        left: sidebarCollapsed ? 80 : 250,
        zIndex: 99,
        transition: 'left 0.2s',
        height: 64,
        lineHeight: '64px',
      }}
    >
      <div className="flex justify-between items-center h-full">
        <div className="flex items-center space-x-4">
          <Text strong className="text-lg text-gray-800">
            服装分类系统
          </Text>
        </div>

        <Space size="large">
          {/* 系统状态 */}
          <Card size="small" className="shadow-sm">
            <Space>
              <Badge
                status={isConnected ? 'success' : 'error'}
                text={
                  <Text type={isConnected ? 'success' : 'danger'}>
                    {isConnected ? '已连接' : '连接断开'}
                  </Text>
                }
              />
              <WifiOutlined className={isConnected ? 'text-green-500' : 'text-red-500'} />
            </Space>
          </Card>

          {/* 模型状态 */}
          {systemStatus && (
            <Card size="small" className="shadow-sm">
              <Space>
                <Text type="secondary">模型:</Text>
                <Space>
                  {systemStatus.current_model ? (
                    <>
                      <CheckCircleOutlined className="text-green-500" />
                      <Text strong>{systemStatus.current_model}</Text>
                    </>
                  ) : (
                    <>
                      <ExclamationCircleOutlined className="text-orange-500" />
                      <Text type="warning">未加载</Text>
                    </>
                  )}
                </Space>
              </Space>
            </Card>
          )}

          {/* GPU状态 */}
          {systemStatus?.gpu_available && (
            <Card size="small" className="shadow-sm">
              <Space>
                <Text type="secondary">GPU:</Text>
                <Badge status="success" text={<Text type="success">可用</Text>} />
              </Space>
            </Card>
          )}
        </Space>
      </div>
    </Header>
  )
}

export default AppHeader

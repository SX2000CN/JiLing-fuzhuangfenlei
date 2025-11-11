import React from 'react'
import { Layout, Menu, Button, Typography } from 'antd'
import {
  FileImageOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined
} from '@ant-design/icons'
import type { MenuProps } from 'antd'

const { Sider } = Layout
const { Title } = Typography

interface SidebarProps {
  collapsed: boolean
  onCollapse: (collapsed: boolean) => void
  selectedKey: string
  onMenuSelect: (key: string) => void
}

const menuItems: MenuProps['items'] = [
  {
    key: 'classification',
    icon: <FileImageOutlined />,
    label: '图像分类',
  },
  {
    key: 'training',
    icon: <ThunderboltOutlined />,
    label: '模型训练',
  },
  {
    key: 'results',
    icon: <BarChartOutlined />,
    label: '结果查看',
  },
  {
    key: 'settings',
    icon: <SettingOutlined />,
    label: '系统设置',
  },
]

const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  onCollapse,
  selectedKey,
  onMenuSelect,
}) => {
  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={onCollapse}
      theme="light"
      width={250}
      style={{
        boxShadow: '2px 0 8px 0 rgba(29, 35, 41, 0.05)',
        borderRight: '1px solid #f0f0f0',
        height: '100vh',
        position: 'fixed',
        left: 0,
        top: 0,
        zIndex: 100,
      }}
    >
      <div className="h-16 flex items-center justify-between px-4 border-b border-gray-200">
        {!collapsed && (
          <Title level={4} className="!mb-0 text-blue-600">
            JiLing 服装分类
          </Title>
        )}
        <Button
          type="text"
          icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          onClick={() => onCollapse(!collapsed)}
          size="small"
        />
      </div>
      
      <Menu
        mode="inline"
        selectedKeys={[selectedKey]}
        items={menuItems}
        onSelect={({ key }) => onMenuSelect(key)}
        style={{
          borderRight: 0,
          height: 'calc(100vh - 64px)',
          paddingTop: '16px',
        }}
      />
    </Sider>
  )
}

export default Sidebar

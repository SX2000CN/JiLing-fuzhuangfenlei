import { useEffect } from 'react'
import { Layout, ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import ClassificationPage from './pages/ClassificationPage'
import TrainingPage from './pages/TrainingPage'
import ResultsPage from './pages/ResultsPage'
import SettingsPage from './pages/SettingsPage'
import { useAppStore } from './store/appStore'
import apiService from './services/api'
import './index.css'

const { Content } = Layout

function App() {
  const { 
    sidebarCollapsed, 
    currentPage, 
    setSidebarCollapsed, 
    setCurrentPage,
    setConnected,
    setSystemStatus
  } = useAppStore()

  // 初始化系统状态
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // 检查API连接状态
        const statusResponse = await apiService.getApiStatus()
        if (statusResponse.success) {
          setConnected(true)
          
          // 获取系统状态
          const systemResponse = await apiService.getSystemStatus()
          if (systemResponse.success && systemResponse.data) {
            setSystemStatus(systemResponse.data)
          }
        }
      } catch (error) {
        console.error('初始化应用失败:', error)
        setConnected(false)
      }
    }

    initializeApp()

    // 定期检查连接状态
    const interval = setInterval(async () => {
      try {
        const response = await apiService.getApiStatus()
        setConnected(response.success)
      } catch {
        setConnected(false)
      }
    }, 10000) // 每10秒检查一次

    return () => clearInterval(interval)
  }, [setConnected, setSystemStatus])

  // 渲染当前页面
  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'classification':
        return <ClassificationPage />
      case 'training':
        return <TrainingPage />
      case 'results':
        return <ResultsPage />
      case 'settings':
        return <SettingsPage />
      default:
        return <ClassificationPage />
    }
  }

  return (
    <ConfigProvider locale={zhCN}>
      <Layout className="min-h-screen">
        <Sidebar
          collapsed={sidebarCollapsed}
          onCollapse={setSidebarCollapsed}
          selectedKey={currentPage}
          onMenuSelect={setCurrentPage}
        />
        
        <Layout
          style={{
            marginLeft: sidebarCollapsed ? 80 : 250,
            transition: 'margin-left 0.2s',
          }}
        >
          <Header sidebarCollapsed={sidebarCollapsed} />
          
          <Content
            style={{
              marginTop: 64,
              minHeight: 'calc(100vh - 64px)',
              background: '#f0f2f5',
            }}
          >
            {renderCurrentPage()}
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  )
}

export default App

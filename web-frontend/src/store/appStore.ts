import { create } from 'zustand'
import type { SystemInfo } from '../types'

interface AppState {
  sidebarCollapsed: boolean
  currentPage: string
  theme: 'light' | 'dark'
  loading: boolean
  
  // 系统状态
  isConnected: boolean
  currentModel: string | null
  systemStatus: SystemInfo | null
  
  setSidebarCollapsed: (collapsed: boolean) => void
  setCurrentPage: (page: string) => void
  setTheme: (theme: 'light' | 'dark') => void
  setLoading: (loading: boolean) => void
  
  // 系统状态操作
  setConnected: (connected: boolean) => void
  setCurrentModel: (model: string | null) => void
  setSystemStatus: (status: SystemInfo | null) => void
}

export const useAppStore = create<AppState>((set) => ({
  sidebarCollapsed: false,
  currentPage: 'classification',
  theme: 'light',
  loading: false,
  
  // 系统状态初始值
  isConnected: false,
  currentModel: null,
  systemStatus: null,
  
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
  setCurrentPage: (page) => set({ currentPage: page }),
  setTheme: (theme) => set({ theme }),
  setLoading: (loading) => set({ loading }),
  
  // 系统状态操作
  setConnected: (connected) => set({ isConnected: connected }),
  setCurrentModel: (model) => set({ currentModel: model }),
  setSystemStatus: (status) => set({ systemStatus: status }),
}))

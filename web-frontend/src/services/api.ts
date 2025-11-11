import axios from 'axios'
import type {
  ApiResponse,
  ClassificationRequest,
  ClassificationResult,
  BatchClassificationResult,
  TrainingConfig,
  TrainingStatus,
  SystemInfo,
  ModelInfo,
  FileInfo
} from '../types'

// 创建 axios 实例
const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加token等
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    console.error('API Error:', error)
    throw error
  }
)

// API 接口定义
export const apiService = {
  // 系统状态
  getSystemStatus: (): Promise<ApiResponse<SystemInfo>> =>
    api.get('/system/status'),

  getApiStatus: (): Promise<ApiResponse> =>
    api.get('/status'),

  // 分类相关
  classifyImage: (data: ClassificationRequest): Promise<ApiResponse<ClassificationResult>> =>
    api.post('/classify', data),

  batchClassifyImages: (data: { folder_path: string; model_name?: string }): Promise<ApiResponse<BatchClassificationResult>> =>
    api.post('/classify/batch', data),

  // 训练相关
  startTraining: (config: TrainingConfig): Promise<ApiResponse> =>
    api.post('/train/start', config),

  stopTraining: (): Promise<ApiResponse> =>
    api.post('/train/stop'),

  getTrainingStatus: (): Promise<ApiResponse<TrainingStatus>> =>
    api.get('/train/status'),

  // 模型相关
  getModels: (): Promise<ApiResponse<ModelInfo[]>> =>
    api.get('/models'),

  loadModel: (model_name: string): Promise<ApiResponse> =>
    api.post('/load_model', { model_name }),

  deleteModel: (model_name: string): Promise<ApiResponse> =>
    api.delete(`/models/${model_name}`),

  // 文件相关
  selectFile: (): Promise<ApiResponse<{ path: string }>> =>
    api.post('/file/select-file'),

  selectFolder: (): Promise<ApiResponse<{ path: string }>> =>
    api.post('/file/select-folder'),

  getFileInfo: (path: string): Promise<ApiResponse<FileInfo>> =>
    api.get(`/file/info?path=${encodeURIComponent(path)}`),

  // 上传文件
  uploadFile: (file: File): Promise<ApiResponse<{ path: string }>> => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },
}

export default apiService

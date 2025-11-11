// API 响应类型
export interface ApiResponse<T = any> {
  success: boolean
  data: T
  message?: string
  error?: string
}

// 分类相关类型
export interface ClassificationRequest {
  image_path: string
  model_name?: string
}

export interface ClassificationResult {
  category: string
  confidence: number
  predictions: Array<{
    category: string
    confidence: number
  }>
}

export interface BatchClassificationResult {
  results: Array<{
    file_name: string
    category: string
    confidence: number
    predictions: Array<{
      category: string
      confidence: number
    }>
  }>
  total: number
  processed: number
}

// 训练相关类型
export interface TrainingConfig {
  model_name: string
  epochs: number
  batch_size: number
  learning_rate: number
  val_split: number
  data_path: string
  pretrained: boolean
  train_mode: string
}

export interface TrainingStatus {
  isTraining: boolean
  progress: number
  currentEpoch: number
  totalEpochs: number
  currentLoss: number
  currentAcc: number
  bestAcc: number
  startTime?: number
  status: string
  logs?: string[]
}

// 系统状态类型
export interface SystemInfo {
  current_model?: string
  gpu_available: boolean
  gpu_memory: {
    used: number
    total: number
  }
  cpu_usage: number
  memory_usage: number
}

// 模型相关类型
export interface ModelInfo {
  name: string
  path: string
  size: number
  created_at: string
  accuracy?: number
  type: string
}

// 文件相关类型
export interface FileInfo {
  name: string
  path: string
  size: number
  type: string
  modified_at: string
}

// 设置相关类型
export interface AppSettings {
  theme: 'light' | 'dark'
  language: 'zh-CN' | 'en-US'
  autoSave: boolean
  defaultModelPath: string
  maxConcurrentTasks: number
}

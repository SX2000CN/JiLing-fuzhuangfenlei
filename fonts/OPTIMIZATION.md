# 字体优化指南

本文档提供 MiSans 字体的性能优化策略，包括子集化、加载优化和降级方案。

## 📊 字体文件体积分析

### 原始 TTF 文件大小

| 字体文件 | 大小（约） | 包含字符 |
|---------|-----------|---------|
| MiSans-Regular.ttf | ~7.7 MB | 完整字符集（含 CJK） |
| MiSans-Medium.ttf | ~7.5 MB | 完整字符集 |
| MiSans-Bold.ttf | ~7.6 MB | 完整字符集 |
| **全部 10 个字重** | **~76 MB** | - |

### 优化后预期大小

| 优化方案 | 单个文件 | 4 个核心字重 | 节省比例 |
|---------|---------|-------------|---------|
| TTF（原始） | ~7.5 MB | ~30 MB | - |
| WOFF2 | ~3 MB | ~12 MB | 60% |
| WOFF2 + 子集化 | ~200-500 KB | ~1-2 MB | **95%+** |

## 🔧 字体子集化指南

### 什么是子集化？

子集化是指从完整字体中提取仅需要使用的字符，创建更小的字体文件。

### 推荐子集化策略

#### 策略一：按语言分割

```
MiSans-Regular.woff2 (完整) → 
├── MiSans-Regular-latin.woff2     (~50 KB)   # 基本拉丁字母
├── MiSans-Regular-chinese.woff2   (~2 MB)    # 常用汉字 6,763 个
└── MiSans-Regular-symbols.woff2   (~30 KB)   # 标点符号
```

#### 策略二：按使用频率

```
├── MiSans-Regular-core.woff2      (~500 KB)  # 高频 3,500 汉字 + 拉丁
├── MiSans-Regular-extended.woff2  (~1.5 MB)  # 扩展汉字
```

#### 策略三：按页面需求

```
├── MiSans-Regular-homepage.woff2  (~100 KB)  # 首页用字
├── MiSans-Regular-article.woff2   (~300 KB)  # 文章页用字
```

### 子集化工具

#### 1. fonttools（Python，推荐）

```bash
# 安装
pip install fonttools brotli

# 子集化为常用汉字 + 拉丁字母
pyftsubset MiSans-Regular.ttf \
  --unicodes="U+0000-007F,U+4E00-9FFF" \
  --output-file="MiSans-Regular-subset.woff2" \
  --flavor=woff2

# 使用字符文件子集化
pyftsubset MiSans-Regular.ttf \
  --text-file=used-characters.txt \
  --output-file="MiSans-Regular-custom.woff2" \
  --flavor=woff2
```

#### 2. glyphhanger（Node.js）

```bash
# 安装
npm install -g glyphhanger

# 分析页面使用的字符
glyphhanger https://example.com --spider --spider-limit=5

# 子集化
glyphhanger --whitelist="你好世界" --subset=MiSans-Regular.ttf --formats=woff2
```

#### 3. 在线工具

- **Font Subsetter**: https://everythingfonts.com/subsetter
- **Fontmin**: http://ecomfe.github.io/fontmin/

### 常用字符集参考

| 字符集 | Unicode 范围 | 字符数 | 用途 |
|-------|-------------|--------|-----|
| 基本拉丁 | U+0000-007F | 128 | 英文、数字 |
| 拉丁扩展 | U+0080-00FF | 128 | 欧洲语言 |
| 中文标点 | U+3000-303F | 64 | 中文标点 |
| 常用汉字 | U+4E00-9FFF | 20,992 | CJK 基本区 |
| GB2312 | 自定义 | 6,763 | 简体中文常用 |

## ⚡ 加载性能优化

### 1. 预加载关键字体

```html
<head>
  <!-- 预加载核心字重 -->
  <link rel="preload" href="/fonts/MiSans-Regular.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="preload" href="/fonts/MiSans-Medium.woff2" as="font" type="font/woff2" crossorigin>
</head>
```

### 2. 使用 font-display 策略

```css
@font-face {
  font-family: "MiSans";
  src: url("/fonts/MiSans-Regular.woff2") format("woff2");
  font-weight: 400;
  font-display: swap;  /* 推荐：先显示后备字体，加载完成后切换 */
}
```

| font-display 值 | 行为 | 适用场景 |
|----------------|------|---------|
| `auto` | 浏览器决定 | 不推荐 |
| `block` | 短暂阻塞后显示 | 品牌字体、图标字体 |
| `swap` | 立即显示后备字体 | **正文字体（推荐）** |
| `fallback` | 短暂阻塞，超时用后备 | 平衡方案 |
| `optional` | 仅缓存时使用 | 极致性能优先 |

### 3. 分阶段加载

```css
/* 首屏只加载 Regular */
@font-face {
  font-family: "MiSans";
  src: url("/fonts/MiSans-Regular.woff2") format("woff2");
  font-weight: 400;
}

/* 其他字重异步加载 */
@font-face {
  font-family: "MiSans";
  src: url("/fonts/MiSans-Bold.woff2") format("woff2");
  font-weight: 700;
}
```

### 4. 使用 Service Worker 缓存

```javascript
// sw.js
const FONT_CACHE = 'fonts-v1';
const FONT_URLS = [
  '/fonts/MiSans-Regular.woff2',
  '/fonts/MiSans-Medium.woff2',
  '/fonts/MiSans-Bold.woff2'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(FONT_CACHE).then((cache) => cache.addAll(FONT_URLS))
  );
});
```

## 🔄 降级方案

### 字体加载失败处理

```css
/* 完整的后备字体栈 */
body {
  font-family: 
    "MiSans",                    /* 首选 */
    -apple-system,               /* macOS/iOS */
    BlinkMacSystemFont,          /* Chrome macOS */
    "PingFang SC",               /* macOS 中文 */
    "Microsoft YaHei",           /* Windows 中文 */
    "Noto Sans SC",              /* Android/Linux */
    sans-serif;                  /* 通用后备 */
}
```

### JavaScript 检测字体加载

```javascript
// 使用 FontFaceObserver 库
const font = new FontFaceObserver('MiSans');

font.load().then(() => {
  document.documentElement.classList.add('fonts-loaded');
}).catch(() => {
  document.documentElement.classList.add('fonts-failed');
  console.warn('MiSans 字体加载失败，使用后备字体');
});
```

```css
/* 根据加载状态调整样式 */
.fonts-loaded body {
  font-family: "MiSans", sans-serif;
}

.fonts-failed body {
  font-family: -apple-system, "PingFang SC", sans-serif;
  /* 可选：调整字间距以匹配 MiSans 视觉效果 */
  letter-spacing: 0.01em;
}
```

## 📋 优化检查清单

- [ ] 字体文件已转换为 WOFF2 格式
- [ ] 仅加载实际使用的字重（建议 ≤ 4 个）
- [ ] 已实施字体子集化（如适用）
- [ ] 已添加 `font-display: swap`
- [ ] 关键字体已预加载
- [ ] 后备字体栈配置完整
- [ ] 字体加载失败有降级方案
- [ ] Service Worker 缓存已配置（如适用）

## 🔗 相关文档

- [字体排版规范](../01-设计基础/04-字体排版.md)
- [字体格式转换指南](./README.md)
- [字体许可证说明](./LICENSE.md)


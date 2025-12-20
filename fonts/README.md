# 字体文件目录

本目录用于存放生产环境使用的 Web 字体文件。

## 📁 目录结构

```
fonts/
├── README.md                    # 本说明文件
├── MiSans-Regular.woff2         # 核心字重
├── MiSans-Medium.woff2
├── MiSans-Semibold.woff2
├── MiSans-Bold.woff2
├── MiSans-Heavy.woff2           # 扩展字重（按需）
├── MiSans-Light.woff2
├── MiSans-ExtraLight.woff2
├── MiSans-Thin.woff2
└── subset/                      # 子集化字体（可选）
    ├── MiSans-Regular-latin.woff2
    └── MiSans-Regular-chinese.woff2
```

## 🔄 字体格式转换指南（TTF → WOFF2）

### 为什么需要转换？

| 格式 | 压缩率 | 浏览器支持 | 推荐程度 |
|------|--------|-----------|---------|
| TTF | 无压缩 | 全部 | 作为后备 |
| WOFF | ~40% | IE9+ | 旧项目 |
| **WOFF2** | **~30%** | **现代浏览器** | **⭐ 推荐** |

MiSans TTF 文件约 7-8MB，转换为 WOFF2 后约 **2-3MB**（节省约 60%）。

### 方法一：在线工具（推荐新手）

#### 1. Transfonter（推荐）
- 网址：https://transfonter.org/
- 步骤：
  1. 上传 TTF 文件
  2. 勾选 "WOFF2" 格式
  3. 点击 "Convert"
  4. 下载转换后的文件

#### 2. Font Squirrel
- 网址：https://www.fontsquirrel.com/tools/webfont-generator
- 步骤：
  1. 上传字体文件
  2. 选择 "Expert" 模式
  3. 仅勾选 WOFF2 格式
  4. 点击 "Download Your Kit"

#### 3. CloudConvert
- 网址：https://cloudconvert.com/ttf-to-woff2
- 支持批量转换

### 方法二：命令行工具（推荐开发者）

#### 使用 Google woff2 工具

```bash
# 1. 安装 woff2（需要先安装）
# macOS
brew install woff2

# Ubuntu/Debian
sudo apt-get install woff2

# Windows (使用 Chocolatey)
choco install woff2

# 2. 转换单个文件
woff2_compress MiSans-Regular.ttf
# 输出: MiSans-Regular.woff2

# 3. 批量转换（Bash）
for file in *.ttf; do woff2_compress "$file"; done

# 批量转换（PowerShell）
Get-ChildItem *.ttf | ForEach-Object { woff2_compress $_.Name }
```

#### 使用 fonttools（Python）

```bash
# 1. 安装
pip install fonttools brotli

# 2. 转换
python -c "from fontTools.ttLib import TTFont; TTFont('MiSans-Regular.ttf').save('MiSans-Regular.woff2')"

# 3. 批量转换脚本
python convert_fonts.py
```

### 方法三：Node.js 工具

```bash
# 安装
npm install -g ttf2woff2

# 使用
cat MiSans-Regular.ttf | ttf2woff2 > MiSans-Regular.woff2

# 批量转换（需要编写脚本）
```

## 📝 文件命名规范

| 原始文件 | 转换后文件 | 说明 |
|---------|-----------|------|
| MiSans-Thin.ttf | MiSans-Thin.woff2 | 保持原名，仅改扩展名 |
| MiSans-Regular.ttf | MiSans-Regular.woff2 | 核心字重 |
| MiSans-Bold.ttf | MiSans-Bold.woff2 | 核心字重 |

**子集化文件命名：**
- `MiSans-Regular-latin.woff2` - 仅含拉丁字符
- `MiSans-Regular-chinese-sc.woff2` - 仅含简体中文
- `MiSans-Regular-subset.woff2` - 自定义子集

## ✅ 转换检查清单

转换完成后，请检查：

- [ ] 所有核心字重已转换：Regular、Medium、Semibold、Bold
- [ ] 文件大小合理（WOFF2 应比 TTF 小 50-70%）
- [ ] 文件命名正确（保持原名，使用 .woff2 扩展名）
- [ ] 在 `examples/font-demo.html` 中测试显示效果
- [ ] 更新 `examples/font-demo.css` 中的字体路径（如需要）

## 🔗 相关文档

- [字体排版规范](../01-设计基础/04-字体排版.md)
- [字体使用示例](../examples/README.md)
- [字体优化指南](./OPTIMIZATION.md)
- [字体许可证说明](./LICENSE.md)


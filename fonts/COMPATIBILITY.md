# 字体兼容性说明

本文档说明 MiSans 字体在不同浏览器、操作系统和设备上的兼容性。

## 🌐 浏览器支持

### WOFF2 格式支持

| 浏览器 | 最低版本 | 发布年份 | 全球占比 |
|--------|---------|---------|---------|
| Chrome | 36+ | 2014 | ~65% |
| Firefox | 39+ | 2015 | ~3% |
| Safari | 12+ | 2018 | ~19% |
| Edge | 14+ | 2016 | ~5% |
| Opera | 23+ | 2014 | ~2% |
| iOS Safari | 12+ | 2018 | ~18% |
| Android Browser | 87+ | 2020 | ~2% |

**总结**: WOFF2 支持覆盖 **95%+** 的现代浏览器用户。

### 不支持 WOFF2 的浏览器

| 浏览器 | 推荐方案 |
|--------|---------|
| IE 11 及以下 | 使用 WOFF 格式作为后备 |
| 旧版 Safari (< 12) | 使用 WOFF 格式作为后备 |
| 旧版 Android (< 5) | 使用 TTF 格式作为后备 |

### 多格式后备方案

```css
@font-face {
  font-family: "MiSans";
  src: url("/fonts/MiSans-Regular.woff2") format("woff2"),  /* 现代浏览器 */
       url("/fonts/MiSans-Regular.woff") format("woff"),    /* IE9+, 旧版Safari */
       url("/fonts/MiSans-Regular.ttf") format("truetype"); /* 旧版Android */
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}
```

## 💻 操作系统兼容性

### 桌面操作系统

| 操作系统 | 支持情况 | 后备字体 |
|---------|---------|---------|
| Windows 10/11 | ✅ 完全支持 | Microsoft YaHei |
| Windows 7/8 | ✅ 支持 | Microsoft YaHei |
| macOS 10.13+ | ✅ 完全支持 | PingFang SC |
| macOS 10.12 及以下 | ✅ 支持 | Hiragino Sans GB |
| Ubuntu/Linux | ✅ 支持 | Noto Sans CJK SC |
| Chrome OS | ✅ 支持 | Noto Sans CJK SC |

### 移动操作系统

| 操作系统 | 支持情况 | 后备字体 |
|---------|---------|---------|
| iOS 12+ | ✅ 完全支持 | PingFang SC |
| iOS 11 及以下 | ⚠️ 部分支持 | PingFang SC |
| Android 5+ | ✅ 支持 | Noto Sans CJK SC |
| HarmonyOS | ✅ 支持 | HarmonyOS Sans SC |

## 🔤 字符集支持

### MiSans 字符覆盖

| 字符集 | 支持情况 | 字符数 |
|-------|---------|--------|
| 基本拉丁 (ASCII) | ✅ 完整 | 128 |
| 拉丁扩展 A/B | ✅ 完整 | 384 |
| 希腊字母 | ✅ 完整 | 134 |
| 西里尔字母 | ✅ 完整 | 256 |
| CJK 基本区 | ✅ 完整 | 20,992 |
| CJK 扩展 A | ✅ 完整 | 6,592 |
| CJK 兼容 | ✅ 完整 | 472 |
| 日文假名 | ✅ 完整 | 186 |
| 韩文音节 | ❌ 不支持 | - |
| 表情符号 | ⚠️ 部分 | - |

### 特殊字符注意事项

| 字符类型 | MiSans 支持 | 建议 |
|---------|------------|------|
| 货币符号 (¥$€) | ✅ 支持 | 正常使用 |
| 数学符号 (±×÷) | ✅ 支持 | 正常使用 |
| 箭头符号 (→←↑↓) | ✅ 支持 | 正常使用 |
| Emoji 表情 | ⚠️ 部分 | 使用系统 Emoji 字体 |
| 生僻汉字 | ⚠️ 部分 | 可能回退到系统字体 |

## 📱 设备特定问题

### iOS 设备

**问题**: iOS 低版本可能出现字体加载延迟

**解决方案**:
```css
/* 针对 iOS 优化 */
@font-face {
  font-family: "MiSans";
  src: url("/fonts/MiSans-Regular.woff2") format("woff2");
  font-weight: 400;
  font-display: swap;
  unicode-range: U+0000-FFFF; /* 限制 Unicode 范围 */
}
```

### Android 设备

**问题**: 部分国产 ROM 可能覆盖 Web 字体

**解决方案**:
- 使用 `!important` 强制应用（不推荐）
- 检测字体加载状态并提供视觉反馈

### 高 DPI 显示器

**问题**: 细体字重在高 DPI 显示器上可能显示过细

**解决方案**:
```css
/* 高 DPI 屏幕使用更粗的字重 */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
  body {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  
  .light-text {
    font-weight: 400; /* 用 Regular 替代 Light */
  }
}
```

## 🔍 兼容性测试清单

部署前请在以下环境测试：

### 必测环境
- [ ] Chrome (Windows)
- [ ] Chrome (macOS)
- [ ] Safari (macOS)
- [ ] Safari (iOS)
- [ ] Chrome (Android)
- [ ] Edge (Windows)

### 建议测试环境
- [ ] Firefox (Windows/macOS)
- [ ] Samsung Internet (Android)
- [ ] 微信内置浏览器 (iOS/Android)
- [ ] 支付宝内置浏览器

### 测试要点
- [ ] 字体正确加载显示
- [ ] 各字重显示正常
- [ ] 中英文混排对齐
- [ ] 加载失败时后备字体正常
- [ ] 无明显的字体闪烁 (FOIT/FOUT)

## 🔗 相关文档

- [字体优化指南](./OPTIMIZATION.md)
- [字体排版规范](../01-设计基础/04-字体排版.md)
- [Can I Use - WOFF2](https://caniuse.com/woff2)


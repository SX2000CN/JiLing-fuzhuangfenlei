#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 修复损坏的字符串
with open('modern_gui_main.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# 查找并替换损坏的字符串
lines = content.split('\n')
for i, line in enumerate(lines):
    if '点击或拖拽图片到此区域上传' in line and 'setText' in line:
        lines[i] = '        self.image_label.setText("📷\\n\\n点击或拖拽图片到此区域上传\\n\\n支持 JPG、PNG、BMP 等格式")'
        print(f"修复第 {i+1} 行: {lines[i]}")

# 写入修复后的内容
with open('modern_gui_main.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("文件修复完成")

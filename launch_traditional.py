#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动传统JiLing服装分类系统
"""
import sys
import os
import subprocess
from pathlib import Path

def kill_previous_instances():
    """关闭之前的应用实例"""
    print("🔍 检查并关闭之前的应用实例...")
    
    try:
        import psutil
        
        # 要关闭的进程关键词
        target_processes = [
            "python.exe gui_main.py",
            "python.exe modern_gui_main.py", 
            "python.exe launch_modern.py",
            "python.exe api_server.py",
            "JiLingClothingClassifier.exe",
            "JiLing服装分类系统.exe"
        ]
        
        killed_count = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                # 检查是否是目标进程
                for target in target_processes:
                    if target.lower() in cmdline.lower():
                        # 避免关闭当前进程
                        if proc.pid != os.getpid():
                            print(f"  ❌ 关闭进程: {proc.info['name']} (PID: {proc.pid})")
                            proc.terminate()
                            killed_count += 1
                            break
                            
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        if killed_count > 0:
            print(f"✅ 已关闭 {killed_count} 个之前的应用实例")
            # 等待进程完全关闭
            import time
            time.sleep(2)
        else:
            print("✅ 没有发现之前的应用实例")
            
    except ImportError:
        print("⚠️ psutil 未安装，使用 taskkill 命令关闭进程...")
        
        # 使用 Windows taskkill 命令作为备选方案
        try:
            # 关闭可能的 Python GUI 进程
            subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                         capture_output=True, check=False)
            
            # 关闭可能的编译后的应用
            subprocess.run(['taskkill', '/f', '/im', 'JiLingClothingClassifier.exe'], 
                         capture_output=True, check=False)
            subprocess.run(['taskkill', '/f', '/im', 'JiLing服装分类系统.exe'], 
                         capture_output=True, check=False)
                         
            print("✅ 已尝试关闭之前的应用实例")
            import time
            time.sleep(1)
            
        except Exception as e:
            print(f"⚠️ 关闭进程时出错: {e}")
    
    except Exception as e:
        print(f"⚠️ 关闭进程时出错: {e}")

def main():
    """启动传统GUI应用"""
    
    # 确保在正确的目录中
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🚀 启动 JiLing 传统服装分类系统...")
    print(f"📁 工作目录: {script_dir}")
    
    # 先关闭之前的实例
    kill_previous_instances()
    
    try:
        print("✅ 启动传统界面...")
        # 运行传统GUI
        subprocess.run([sys.executable, "gui_main.py"], check=True)
        
    except Exception as e:
        print(f"❌ 启动传统应用时出错: {e}")
        print("\n💡 请检查以下依赖是否已安装:")
        print("   - PySide6: pip install PySide6")
        print("   - PyTorch: pip install torch torchvision")
        print("   - 其他依赖: pip install -r requirements.txt")

if __name__ == "__main__":
    main()

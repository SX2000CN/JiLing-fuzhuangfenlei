#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JiLing 服装分类系统 - 统一启动器
支持现代化版本和传统版本的选择启动
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
            "python.exe launch_traditional.py",
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

def launch_modern_gui():
    """启动现代化GUI"""
    try:
        from modern_gui_main import main as run_modern_gui
        print("✅ 加载现代化界面...")
        run_modern_gui()
    except ImportError as e:
        print(f"❌ 无法加载现代化界面: {e}")
        return False
    except Exception as e:
        print(f"❌ 启动现代化界面出错: {e}")
        return False
    return True

def launch_traditional_gui():
    """启动传统GUI"""
    try:
        print("✅ 启动传统界面...")
        subprocess.run([sys.executable, "gui_main.py"], check=True)
        return True
    except Exception as e:
        print(f"❌ 启动传统界面出错: {e}")
        return False

def main():
    """主启动函数"""
    
    # 确保在正确的目录中
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🚀 JiLing 服装分类系统启动器")
    print(f"📁 工作目录: {script_dir}")
    print("=" * 50)
    
    # 先关闭之前的实例
    kill_previous_instances()
    print("=" * 50)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['modern', 'm', '1']:
            print("🎨 启动现代化版本...")
            if not launch_modern_gui():
                print("🔄 回退到传统版本...")
                launch_traditional_gui()
        elif sys.argv[1].lower() in ['web', 'w']:
            print("🌐 启动 Web 外壳版本...")
            try:
                subprocess.run([sys.executable, "web_shell.py"], check=True)
            except Exception as e:
                print(f"❌ 启动 Web 外壳失败: {e}")
        elif sys.argv[1].lower() in ['traditional', 't', '2']:
            print("🏛️ 启动传统版本...")
            launch_traditional_gui()
        else:
            print(f"❌ 无效参数: {sys.argv[1]}")
            print("💡 使用方法: python launch.py [web|modern|traditional]")
    else:
        # 没有参数时，显示选择菜单
        print("请选择要启动的版本:")
        print("1. 🌐 Web 外壳版本 (像素级一致)")
        print("2. 🎨 现代化版本 (PySide6 原生)")
        print("3. 🏛️ 传统版本")
        print("4. ❌ 退出")
        
        while True:
            try:
                choice = input("\n请输入选择 (1-3): ").strip()
                
                if choice == '1':
                    print("� 启动 Web 外壳版本...")
                    try:
                        subprocess.run([sys.executable, "web_shell.py"], check=True)
                    except Exception as e:
                        print(f"❌ 启动 Web 外壳失败: {e}")
                    break
                elif choice == '2':
                    print("�🎨 启动现代化版本...")
                    if not launch_modern_gui():
                        print("🔄 回退到传统版本...")
                        launch_traditional_gui()
                    break
                elif choice == '3':
                    print("🏛️ 启动传统版本...")
                    launch_traditional_gui()
                    break
                elif choice == '4':
                    print("👋 退出启动器")
                    break
                else:
                    print("❌ 请输入 1、2、3 或 4")
                    
            except KeyboardInterrupt:
                print("\n👋 退出启动器")
                break
            except Exception as e:
                print(f"❌ 输入错误: {e}")

if __name__ == "__main__":
    main()

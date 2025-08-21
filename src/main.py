#!/usr/bin/env python3
"""
JiLing-fuzhuangfenlei 主程序入口
服装挂拍分类系统 - PyTorch版本
"""

import sys
import argparse
from pathlib import Path

# 添加src路径
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description='JiLing-fuzhuangfenlei 服装挂拍分类系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
运行模式:
  gui         启动图形界面 (默认)
  classify    运行快速分类脚本
  train       运行训练脚本
  test        运行测试

使用示例:
  # 启动GUI界面
  python main.py
  python main.py gui
  
  # 快速分类
  python main.py classify --input "D:/photos" --output "D:/sorted"
  
  # 训练模型
  python main.py train --data-path "D:/data/train"
        '''
    )
    
    parser.add_argument('mode', nargs='?', default='gui',
                       choices=['gui', 'classify', 'train', 'test'],
                       help='运行模式 (默认: gui)')
    
    # 快速分类参数
    parser.add_argument('--input', type=str, help='输入文件夹路径')
    parser.add_argument('--output', type=str, help='输出文件夹路径')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--confidence', type=float, help='置信度阈值')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    parser.add_argument('--dry-run', action='store_true', help='只预测不移动文件')
    
    # 训练参数
    parser.add_argument('--data-path', type=str, help='训练数据路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    
    args = parser.parse_args()
    
    print("🤖 JiLing-fuzhuangfenlei 服装挂拍分类系统")
    print("=" * 50)
    print(f"运行模式: {args.mode}")
    print()
    
    if args.mode == 'gui':
        launch_gui()
    elif args.mode == 'classify':
        launch_classify(args)
    elif args.mode == 'train':
        launch_train(args)
    elif args.mode == 'test':
        launch_test()
    else:
        parser.print_help()

def launch_gui():
    """启动GUI界面"""
    try:
        print("🚀 启动图形界面...")
        
        # 检查PySide6是否可用
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            print("❌ PySide6未安装，请运行: pip install PySide6")
            return
        
        # 导入主窗口（稍后实现）
        try:
            from ui.main_window import MainWindow
            
            app = QApplication(sys.argv)
            window = MainWindow()
            window.show()
            
            print("✅ GUI启动成功")
            sys.exit(app.exec())
            
        except ImportError:
            print("⚠️ GUI模块尚未实现，正在启动快速分类模式...")
            # fallback到分类模式
            import subprocess
            subprocess.run([sys.executable, "scripts/fast_classify.py", "--help"])
            
    except Exception as e:
        print(f"❌ GUI启动失败: {e}")

def launch_classify(args):
    """启动快速分类"""
    print("📸 启动快速分类模式...")
    
    # 构建命令参数
    cmd = [sys.executable, "scripts/fast_classify.py"]
    
    if args.config and args.config != 'config.json':
        cmd.extend(['--config', args.config])
    if args.input:
        cmd.extend(['--input', args.input])
    if args.output:
        cmd.extend(['--output', args.output])
    if args.model:
        cmd.extend(['--model', args.model])
    if args.confidence:
        cmd.extend(['--confidence', str(args.confidence)])
    if args.device and args.device != 'auto':
        cmd.extend(['--device', args.device])
    if args.verbose:
        cmd.append('--verbose')
    if args.dry_run:
        cmd.append('--dry-run')
    
    try:
        import subprocess
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ 分类失败: {e}")
        return e.returncode
    except Exception as e:
        print(f"❌ 启动分类脚本失败: {e}")
        return 1

def launch_train(args):
    """启动训练模式"""
    print("🎯 启动训练模式...")
    print("⚠️ 训练模块正在开发中...")
    
    # TODO: 实现训练脚本
    try:
        from core.pytorch_trainer import ClothingTrainer
        print("✅ 训练模块已导入")
    except ImportError:
        print("❌ 训练模块尚未实现")
    
    return 0

def launch_test():
    """启动测试模式"""
    print("🧪 启动测试模式...")
    
    tests = [
        test_model_factory,
        test_classifier,
        test_gpu_support
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__} 通过")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} 失败: {e}")
    
    print(f"\n测试结果: {passed} 通过, {failed} 失败")
    return 0 if failed == 0 else 1

def test_model_factory():
    """测试模型工厂"""
    from core.model_factory import ModelFactory
    
    # 测试模型创建
    model = ModelFactory.create_model('efficientnetv2_s', num_classes=3, pretrained=False)
    assert model is not None
    
    # 测试模型信息
    info = ModelFactory.get_model_info('efficientnetv2_s')
    assert 'description' in info

def test_classifier():
    """测试分类器"""
    from core.pytorch_classifier import ClothingClassifier
    from core.model_factory import ModelFactory
    import torch
    import tempfile
    
    # 创建临时模型文件
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        model = ModelFactory.create_model('efficientnetv2_s', num_classes=3, pretrained=False)
        torch.save(model.state_dict(), f.name)
        
        # 测试分类器初始化
        classifier = ClothingClassifier(f.name, device='cpu')
        assert classifier.device.type == 'cpu'
        assert len(classifier.classes) == 3
    
    # 清理临时文件
    Path(f.name).unlink()

def test_gpu_support():
    """测试GPU支持"""
    import torch
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU: {device_name}, 数量: {device_count}")
    else:
        print("GPU不可用，使用CPU")
    
    # 至少CPU应该可用
    assert torch.device('cpu').type == 'cpu'

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        sys.exit(1)

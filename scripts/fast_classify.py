#!/usr/bin/env python3
"""
服装图片快速分类脚本
独立运行的命令行工具，用于批量分类服装图片
"""

import argparse
import json
import sys
from pathlib import Path
import logging
import time

# 添加项目路径到sys.path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from core.pytorch_classifier import ClothingClassifier
    from core.model_factory import ModelFactory
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fast_classify.log', encoding='utf-8')
        ]
    )

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        # 创建默认配置文件
        default_config = {
            "model_config": {
                "model_name": "efficientnetv2_s",
                "model_path": "models/saved_models/best_model.pth",
                "num_classes": 3
            },
            "paths": {
                "input_folder": "D:/桌面/筛选/JPG",
                "output_folder": "D:/桌面/筛选",
                "log_folder": "logs"
            },
            "classification": {
                "batch_size": 32,
                "confidence_threshold": 0.5,
                "classes": ["主图", "细节", "吊牌"]
            },
            "processing": {
                "move_files": True,
                "save_statistics": True,
                "create_subfolders": True
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已创建默认配置文件: {config_path}")
        return default_config
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def validate_paths(config: dict, args: argparse.Namespace) -> tuple:
    """验证和获取路径"""
    # 输入路径
    input_folder = args.input or config['paths']['input_folder']
    input_path = Path(input_folder)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
    
    # 输出路径
    output_folder = args.output or config['paths']['output_folder']
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 模型路径
    model_path = args.model or config['model_config']['model_path']
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    return str(input_path), str(output_path), model_path

def print_header():
    """打印程序头部信息"""
    print("🤖 JiLing-fuzhuangfenlei 快速分类工具")
    print("=" * 50)
    print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_config_info(config: dict, input_folder: str, output_folder: str, model_path: str):
    """打印配置信息"""
    model_name = config['model_config']['model_name']
    confidence = config['classification']['confidence_threshold']
    move_files = config['processing']['move_files']
    
    print("📋 配置信息:")
    print(f"  输入文件夹: {input_folder}")
    print(f"  输出文件夹: {output_folder}")
    print(f"  模型文件: {model_path}")
    print(f"  模型类型: {model_name}")
    print(f"  置信度阈值: {confidence}")
    print(f"  文件处理: {'移动' if move_files else '复制'}")
    print()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='服装图片快速分类工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 使用默认配置
  python fast_classify.py
  
  # 使用自定义参数
  python fast_classify.py --input "D:/photos" --output "D:/sorted" --confidence 0.8
  
  # 使用自定义配置文件
  python fast_classify.py --config my_config.json --verbose
        '''
    )
    
    parser.add_argument('--config', type=str, default='config.json', 
                       help='配置文件路径 (默认: config.json)')
    parser.add_argument('--input', type=str, 
                       help='输入文件夹路径 (覆盖配置文件)')
    parser.add_argument('--output', type=str, 
                       help='输出文件夹路径 (覆盖配置文件)')
    parser.add_argument('--model', type=str, 
                       help='模型路径 (覆盖配置文件)')
    parser.add_argument('--confidence', type=float, 
                       help='置信度阈值 (0.0-1.0)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='计算设备 (默认: auto)')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细日志')
    parser.add_argument('--dry-run', action='store_true',
                       help='只预测不移动文件')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    try:
        print_header()
        
        # 加载配置
        config = load_config(args.config)
        
        # 验证路径
        input_folder, output_folder, model_path = validate_paths(config, args)
        
        # 打印配置信息
        print_config_info(config, input_folder, output_folder, model_path)
        
        # 更新配置参数
        if args.confidence:
            config['classification']['confidence_threshold'] = args.confidence
        
        if args.dry_run:
            config['processing']['move_files'] = False
            print("🔍 DRY RUN 模式: 只预测不移动文件")
            print()
        
        # 初始化分类器
        print("🚀 初始化分类器...")
        model_name = config['model_config']['model_name']
        classifier = ClothingClassifier(
            model_path=model_path,
            device=args.device,
            model_name=model_name
        )
        print("✅ 分类器初始化完成")
        print()
        
        # 开始分类
        print("📸 开始图片分类...")
        start_time = time.time()
        
        results = classifier.classify_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            confidence_threshold=config['classification']['confidence_threshold'],
            move_files=config['processing']['move_files'] and not args.dry_run,
            save_results=config['processing']['save_statistics']
        )
        
        # 显示最终结果
        print()
        print("🎉 分类完成!")
        print(f"⏱️ 总耗时: {time.time() - start_time:.2f}秒")
        print(f"📊 成功率: {results['processed']/results['total']*100:.1f}%")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        return 1
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

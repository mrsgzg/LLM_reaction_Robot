#!/usr/bin/env python3
"""
检查 Reaction_Dataset/train 下不同模态数据的对应关系
验证 audio, video-face-crop, AU_Continue, coefficients, facial-attributes 等模态数据是否一一对应
"""

import os
from pathlib import Path
from collections import defaultdict
import argparse


def get_file_stem(filepath, modality_name=None):
    """
    获取文件名（不含扩展名）
    对于AU_Continue模态，会移除_AUs后缀以便匹配
    """
    stem = Path(filepath).stem
    
    # 如果是AU_Continue模态，移除_AUs后缀
    if modality_name == 'AU_Continue' and stem.endswith('_AUs'):
        stem = stem[:-4]  # 移除最后4个字符 '_AUs'
    
    return stem


def scan_modality_files(modality_path, role, modality_name=None):
    """
    扫描某个模态下的所有文件
    返回: {session_name: [file_stems]}
    """
    session_files = defaultdict(list)
    role_path = modality_path / role
    
    if not role_path.exists():
        return session_files
    
    for session_dir in sorted(role_path.iterdir()):
        if session_dir.is_dir():
            session_name = session_dir.name
            files = []
            for file in sorted(session_dir.iterdir()):
                if file.is_file():
                    files.append(get_file_stem(file.name, modality_name))
            session_files[session_name] = files
    
    return session_files


def check_correspondence(train_path):
    """检查各个模态数据的对应关系"""
    
    train_path = Path(train_path)
    
    # 定义所有模态
    modalities = ['audio', 'video-face-crop', 'AU_Continue', 'coefficients', 'facial-attributes']
    roles = ['listener', 'speaker']
    
    print("="*80)
    print("开始检查 Reaction_Dataset/train 下的数据对应关系")
    print("="*80)
    print()
    
    # 存储每个模态的文件信息
    modality_data = {}
    
    # 扫描所有模态
    for modality in modalities:
        modality_path = train_path / modality
        if not modality_path.exists():
            print(f"⚠️  警告: 模态文件夹不存在: {modality}")
            continue
        
        print(f"📂 扫描模态: {modality}")
        modality_data[modality] = {}
        
        for role in roles:
            files = scan_modality_files(modality_path, role, modality)
            modality_data[modality][role] = files
            total_files = sum(len(files_list) for files_list in files.values())
            print(f"   └─ {role}: {len(files)} sessions, {total_files} files")
        print()
    
    # 检查对应关系
    print("="*80)
    print("检查数据对应关系")
    print("="*80)
    print()
    
    all_issues = []
    
    for role in roles:
        print(f"\n🔍 检查 {role.upper()} 的数据对应关系:")
        print("-" * 80)
        
        # 获取所有session的并集
        all_sessions = set()
        for modality in modalities:
            if modality in modality_data:
                all_sessions.update(modality_data[modality].get(role, {}).keys())
        
        for session in sorted(all_sessions):
            session_issues = []
            
            # 检查每个模态是否有这个session
            modality_files = {}
            for modality in modalities:
                if modality not in modality_data:
                    continue
                    
                role_data = modality_data[modality].get(role, {})
                if session in role_data:
                    modality_files[modality] = set(role_data[session])
                else:
                    session_issues.append(f"  ⚠️  缺失整个session: {modality}")
            
            # 检查文件对应关系
            if len(modality_files) > 1:
                # 使用第一个模态作为参考
                reference_modality = list(modality_files.keys())[0]
                reference_files = modality_files[reference_modality]
                
                for modality, files in modality_files.items():
                    if modality == reference_modality:
                        continue
                    
                    # 检查缺失的文件
                    missing = reference_files - files
                    extra = files - reference_files
                    
                    if missing:
                        session_issues.append(f"  ⚠️  {modality} 缺失文件 ({len(missing)}): {', '.join(sorted(list(missing))[:5])}{'...' if len(missing) > 5 else ''}")
                    
                    if extra:
                        session_issues.append(f"  ⚠️  {modality} 多余文件 ({len(extra)}): {', '.join(sorted(list(extra))[:5])}{'...' if len(extra) > 5 else ''}")
            
            # 报告结果
            if session_issues:
                print(f"\n❌ {session}:")
                for issue in session_issues:
                    print(issue)
                    all_issues.append(f"{role}/{session}: {issue.strip()}")
            else:
                if modality_files:
                    file_count = len(list(modality_files.values())[0])
                    print(f"✅ {session}: {file_count} files, 所有模态数据对应正确")
    
    # 总结
    print("\n" + "="*80)
    print("检查总结")
    print("="*80)
    
    if all_issues:
        print(f"\n❌ 发现 {len(all_issues)} 个问题:")
        for issue in all_issues[:20]:  # 只显示前20个问题
            print(f"  - {issue}")
        if len(all_issues) > 20:
            print(f"  ... 还有 {len(all_issues) - 20} 个问题未显示")
    else:
        print("\n✅ 所有数据对应关系正确！")
    
    return len(all_issues) == 0


def main():
    parser = argparse.ArgumentParser(description='检查 Reaction_Dataset 中不同模态数据的对应关系')
    parser.add_argument('--path', type=str, 
                       default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/LLM_reaction_Robot/Reaction_DataSet/train',
                       help='train 文件夹路径')
    
    args = parser.parse_args()
    
    train_path = Path(args.path)
    
    if not train_path.exists():
        print(f"❌ 错误: 路径不存在: {train_path}")
        return 1
    
    success = check_correspondence(train_path)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

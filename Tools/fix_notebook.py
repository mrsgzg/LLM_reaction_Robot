#!/usr/bin/env python3
"""
Script to fix data_analysis.ipynb:
1. Convert all Chinese text to English
2. Fix JSON serialization issues with numpy types
"""

import re
import sys

# Read the notebook
with open('data_analysis.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Chinese to English translations mapping
translations = {
    # Comments and strings
    '配置matplotlib为无GUI模式（服务器环境）': 'Configure matplotlib for no-GUI mode (server environment)',
    '设置绘图风格': 'Set plotting style',
    '导入必要的库': 'Import necessary libraries',
    '加载数据集': 'Load datasets',
    '分析每个AU的激活频率': 'Analyze activation frequency for each AU',
    '统计每个AU的激活次数': 'Count activations for each AU',
    '计算激活率': 'Calculate activation rates',
    '创建DataFrame便于分析': 'Create DataFrame for analysis',
    '排序': 'Sort by activation rate',
    'AU激活统计（按频率排序）：': 'AU Activation Statistics (sorted by frequency):',
    '频率分类：': 'Frequency Classification:',
    '高频AU': 'High-freq AU',
    '中频AU': 'Mid-freq AU',  
    '低频AU': 'Low-freq AU',
    '不平衡比（最高/最低）': 'Imbalance Ratio (max/min)',
    '分析训练集和验证集': 'Analyze train and val sets',
    '可视化AU激活频率': 'Visualize AU activation frequency',
    '训练集': 'Train Set',
    '验证集': 'Val Set',
    '保存图表': 'Saved figure',
    '计算建议的损失权重（基于逆频率）': 'Calculate recommended loss weights (based on inverse frequency)',
    '基于激活率计算损失权重：低频AU获得更高权重': 'Calculate loss weights based on activation rates: lower-freq AUs get higher weights',
    '方法1: 逆频率（加平滑）': 'Method 1: Inverse frequency (with smoothing)',
    'log平滑避免极端值': 'log smoothing to avoid extreme values',
    '归一化到\\[0.5, 2.0\\]范围': 'Normalize to [0.5, 2.0] range',
    '建议的AU损失权重（用于Focal Loss或Weighted BCE）：': 'Recommended AU Loss Weights (for Focal Loss or Weighted BCE):',
    '激活率': 'Activation Rate',
    '权重': 'Weight',
    '保存权重配置': 'Saved weight config',
    '分析speaker-listener的反应延迟模式': 'Analyze speaker-listener reaction delay patterns',
    '方法：计算listener AU激活的时间相对于speaker视频开始的延迟': 'Method: Calculate listener AU activation time delay relative to speaker video start',
    '这是简化版分析，完整版需要解析speaker的动作时间戳': 'This is simplified analysis, complete version needs to parse speaker action timestamps',
    '每个AU首次激活的延迟': 'Delay of first activation for each AU',
    'AU状态变化的间隔': 'Intervals between AU state changes',
    '采样分析（完整分析会很慢）': 'Sample analysis (full analysis would be slow)',
    '找到首次激活的帧': 'Find first activation frame',
    '找到状态变化点（0→1或1→0）': 'Find state change points (0→1 or 1→0)',
    '统计结果': 'Statistical results',
    'AU首次激活延迟统计（相对于视频开始）：': 'AU First Activation Delay Statistics (relative to video start):',
    '平均': 'Mean',
    '中位数': 'Median',
    '标准差': 'Std Dev',
    '样本数': 'N Samples',
    '分析训练集': 'Analyze train set',
    '可视化延迟分布': 'Visualize delay distribution',
    'AU首次激活延迟的箱线图': 'AU First Activation Delay Box Plot',
    'AU首次激活延迟分布（相对于视频开始）': 'AU First Activation Delay Distribution (relative to video start)',
    '平均延迟的条形图': 'Average Delay Bar Chart',
    '各AU的平均首次激活延迟': 'Mean First Activation Delay by AU',
    '参考线': 'Reference line',
    '全局延迟分布直方图': 'Global Delay Distribution Histogram',
    '所有AU的首次激活延迟分布': 'First Activation Delay Distribution for All AUs',
    '建议的时间偏移量': 'Recommended Time Offset',
    '延迟分析结果与建议': 'Delay Analysis Results and Recommendations',
    '全局统计：': 'Global Statistics:',
    '中位数延迟': 'Median Delay',
    '平均延迟': 'Mean Delay',
    '训练建议：': 'Training Recommendations:',
    '时间偏移设置（τ）：': 'Time Offset Setting (τ):',
    '推荐': 'Recommended',
    '帧': 'frames',
    '训练代码：': 'Training Code:',
    '不同策略对比：': 'Strategy Comparison:',
    '无偏移': 'No offset',
    '学习同步反应（不自然）': 'Learn synchronous reaction (unnatural)',
    '固定偏移': 'Fixed offset',
    '学习平均延迟': 'Learn average delay',
    '自适应偏移': 'Adaptive offset',
    '为每个AU设置不同τ': 'Different τ for each AU',
    '机器人部署：': 'Robot Deployment:',
    '预测未来': 'Predict future',
    '的AU': 'AU',
    '可补偿网络\\+机械延迟': 'Can compensate network + mechanical delay',
    '保存延迟配置': 'Saved delay config',
    '分析样本长度分布': 'Analyze sample length distribution',
    '样本数量': 'Number of samples',
    '帧数统计:': 'Frame Statistics:',
    '最小': 'Min',
    '最大': 'Max',
    '百分位数:': 'Percentiles:',
    '秒': 'sec',
    '分析两个集合': 'Analyze both sets',
    '可视化长度分布': 'Visualize length distribution',
    '帧数分布直方图': 'Frame Count Distribution Histogram',
    '样本帧数分布': 'Sample Frame Count Distribution',
    '累积分布函数（CDF）': 'Cumulative Distribution Function (CDF)',
    '箱线图对比': 'Box Plot Comparison',
    '帧数分布箱线图': 'Frame Count Distribution Box Plot',
    '建议的配置': 'Recommended Configuration',
    '计算建议的max_seq_length': 'Calculate recommended max_seq_length',
    '样本长度分析结果与建议': 'Sample Length Analysis Results and Recommendations',
    '统计摘要（Train）:': 'Statistical Summary (Train):',
    '平均长度': 'Mean Length',
    '样本': 'samples',
    '训练配置建议：': 'Training Configuration Recommendations:',
    '方案A（覆盖90%）': 'Option A (90% coverage)',
    '方案B（覆盖95%）': 'Option B (95% coverage)',
    '方案C（覆盖99%）': 'Option C (99% coverage)',
    '内存优化策略:': 'Memory Optimization Strategy:',
    '如选择方案B': 'If choosing Option B',
    '可节省': 'Can save',
    '显存': 'GPU memory',
    '仅丢弃': 'Only discard',
    '最长样本': 'longest samples',
    '或对长样本裁剪/滑动窗口': 'Or crop/sliding window for long samples',
    '滑动窗口训练:': 'Sliding Window Training:',
    '窗口大小': 'Window size',
    '步长': 'Stride',
    '重叠训练': 'overlapping training',
    '估算:': 'Estimation:',
    '假设': 'Assuming',
    '保存长度配置': 'Saved length config',
    '全面的数据质量检查': 'Comprehensive data quality check',
    '检查': 'Check',
    '集的数据质量': 'set data quality',
    '检查文件存在性': 'Check file existence',
    '检查AU概率范围': 'Check AU probability range',
    '检查帧数一致性': 'Check frame count consistency',
    '容差±0.1秒': 'tolerance ±0.1 sec',
    '一致性': 'consistency',
    '应该都是30 FPS': 'should all be 30 FPS',
    '检查过短样本': 'Check samples that are too short',
    '检查AU标注噪声（抖动）': 'Check AU annotation noise (jitter)',
    '计算变化频率': 'Calculate change frequency',
    '的帧在变化（可能太抖）': 'of frames changing (possibly too jittery)',
    '报告结果': 'Report results',
    '质量检查结果:': 'Quality Check Results:',
    '显示前3个例子': 'Show first 3 examples',
    '分布': 'distribution',
    '检查两个集合': 'Check both sets',
    '检查train和val之间的数据泄漏': 'Check data leakage between train and val',
    '检查数据泄漏（Train-Val重叠）': 'Check Data Leakage (Train-Val Overlap)',
    '警告: 发现': 'Warning: Found',
    '个重复样本ID！': 'duplicate sample IDs!',
    '重复ID示例': 'Duplicate ID examples',
    '这会导致数据泄漏，验证指标虚高。建议清理数据。': 'This causes data leakage, inflating validation metrics. Recommend cleaning data.',
    '无重复样本，train和val完全分离。': 'No duplicate samples, train and val are completely separated.',
    '生成数据质量报告': 'Generate data quality report',
    '保存质量报告': 'Saved quality report',
    '生成综合建议报告': 'Generate comprehensive recommendation report',
    '数据分析完成！综合建议：': 'Data analysis complete! Comprehensive recommendations:',
    '数据集分析摘要报告': 'Dataset Analysis Summary Report',
    '生成时间': 'Generated at',
    '数据集规模': 'Dataset Size',
    'AU激活频率（关键发现）': 'AU Activation Frequency (Key Findings)',
    '建议: 使用加权损失函数（配置已保存到 au_loss_weights.json）': 'Recommendation: Use weighted loss function (config saved to au_loss_weights.json)',
    '反应延迟模式': 'Reaction Delay Pattern',
    '推荐帧偏移': 'Recommended frame offset',
    '建议: 训练时使用': 'Recommendation: Use during training',
    '样本长度分布': 'Sample Length Distribution',
    '建议 max_seq_length': 'Recommended max_seq_length',
    '覆盖95%样本': 'covers 95% samples',
    '或使用滑动窗口': 'or use sliding window',
    '窗口': 'window',
    '数据质量': 'Data Quality',
    '缺失文件': 'Missing files',
    '无效AU概率': 'Invalid AU probs',
    '数据泄漏': 'Data leakage',
    '重复样本': 'duplicate samples',
    '状态': 'Status',
    '需要清理数据': 'Need to clean data',
    '数据质量良好': 'Data quality good',
    '下一步行动': 'Next Actions',
    '查看生成的图表': 'View generated figures',
    '使用配置文件:': 'Use config files:',
    '损失权重': 'loss weights',
    '时间偏移': 'time offset',
    '序列长度': 'sequence length',
    '开始训练前建议:': 'Before starting training:',
    '如有数据问题，先清理': 'If data issues exist, clean first',
    '根据max_seq_length调整batch_size': 'Adjust batch_size based on max_seq_length',
    '在训练脚本中应用AU权重和延迟偏移': 'Apply AU weights and delay offset in training script',
    '保存摘要报告': 'Saved summary report',
    '所有分析完成！生成的文件：': 'All analysis complete! Generated files:',
    
    # Markdown headers
    'AU激活频率分布分析': 'AU Activation Frequency Distribution Analysis',
    '目的': 'Objectives',
    '识别高频AU vs 低频AU': 'Identify high-freq vs low-freq AUs',
    '计算不平衡比例': 'Calculate imbalance ratio',
    '为损失函数设计权重': 'Design weights for loss function',
    'Speaker-Listener延迟模式分析': 'Speaker-Listener Delay Pattern Analysis',
    '发现听众反应的自然延迟': 'Discover natural listener reaction delays',
    '为时间偏移训练提供数据依据': 'Provide data basis for time-offset training',
    '不同AU可能有不同的反应延迟': 'Different AUs may have different reaction delays',
    '了解序列长度分布': 'Understand sequence length distribution',
    '优化batch size和max_seq_length': 'Optimize batch size and max_seq_length',
    '设计训练窗口策略': 'Design training window strategy',
    '数据质量检查': 'Data Quality Check',
    '检查文件完整性': 'Check file integrity',
    '发现异常值和噪声': 'Discover outliers and noise',
    '总结与建议': 'Summary and Recommendations',
    '基于以上分析，生成最终的配置建议': 'Based on the above analysis, generate final configuration recommendations',
}

# Apply all translations
for chinese, english in translations.items():
    content = content.replace(chinese, english)

# Add JSON serialization fix to the delay config section
content = content.replace(
    "delay_config = {\n    'recommended_delay_ms': float(global_median),",
    "delay_config = convert_to_native_types({\n    'recommended_delay_ms': float(global_median),"
)

content = content.replace(
    "    'per_au_delays': train_delay_stats\n}",
    "    'per_au_delays': train_delay_stats\n})"
)

# Add JSON serialization fix to the length config section  
content = content.replace(
    "length_config = {\n    'train_statistics': train_length_stats,",
    "length_config = convert_to_native_types({\n    'train_statistics': train_length_stats,"
)

content = content.replace(
    "        'sliding_window_stride': 500\n    }\n}\nwith open(output_dir / 'sample_length_config.json', 'w') as f:",
    "        'sliding_window_stride': 500\n    }\n})\nwith open(output_dir / 'sample_length_config.json', 'w') as f:"
)

# Write back
with open('data_analysis.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed notebook: converted Chinese to English and added JSON serialization fixes")

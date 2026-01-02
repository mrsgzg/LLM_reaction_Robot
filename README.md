# Speaker-to-Listener Reaction Generation

## 项目概述

本项目旨在开发一个基于说话者多模态输入（视频+音频）预测听众面部反应的智能系统，最终目标是驱动机器人平台实现实时、自然的非言语反应生成。

---

## 核心任务

**任务定义**：给定说话者（Speaker）的视频和音频输入，实时预测听众（Listener）的面部动作单元（Action Units, AU）变化序列。

**任务类型**：
- 跨模态序列到序列映射（视觉+听觉 → AU序列）
- 跨人物反应生成（说话者 → 听众）
- 时序预测与生成

**应用场景**：
- 社交机器人的非言语交互
- 虚拟代理的情感反应生成
- 人机对话系统的共情能力增强

---

## 数据集说明

### 基本信息

- **数据路径**：`/net/scratch/k09562zs/LLM_reaction_Robot/Reaction_DataSet/`
- **数据规模**：
  - **训练集**：1,660个样本对
  - **验证集**：571个样本对
- **组织结构**：会话级别（session-based），每个会话包含时间同步的speaker-listener配对数据

### 样本数据结构

每个样本包含以下字段：

```python
{
    # 样本标识
    'id': str,  # 格式："session_name_timestamp"
    
    # === 输入：说话者（Speaker）多模态数据 ===
    'speaker_video_path': str,  # 说话者视频路径（MP4，30 FPS）
    'speaker_audio_path': str,  # 说话者音频路径（WAV）
    
    # === 目标：听众（Listener）面部反应标注 ===
    'listener_video_path': str,      # 听众视频路径（用于参考/验证）
    'listener_audio_path': str,      # 听众音频路径（用于参考）
    'listener_au_names': List[str],  # AU名称列表：['AU1', 'AU2', ...]
    'listener_au_prob': Dict[str, List[float]],  # 每个AU的概率序列（0.0-1.0）
    'listener_au_act': Dict[str, List[int]],     # 每个AU的激活序列（0或1）
    'listener_frame_idx': List[int],  # 帧索引序列
    
    # === 元数据 ===
    'fps': float,        # 视频帧率（统一为30 FPS）
    'duration': float,   # 视频时长（秒）
    'n_frames': int      # 总帧数
}
```

### AU（面部动作单元）标注详情

- **AU数量**：17个面部动作单元
- **AU列表**：
  ```
  AU1  (眉毛内侧上扬)    AU2  (眉毛外侧上扬)    AU4  (眉毛下压)
  AU5  (上眼睑提升)      AU6  (脸颊提升)        AU7  (眼睑收紧)
  AU9  (鼻子皱起)        AU10 (上唇提升)        AU12 (嘴角上扬)
  AU14 (酒窝)            AU15 (嘴角下压)        AU17 (下巴上提)
  AU18 (嘴唇撅起)        AU20 (嘴唇横向拉伸)    AU23 (嘴唇收紧)
  AU25 (嘴唇分开)        AU26 (下巴下垂)
  ```

- **标注形式**：
  - `au_prob`：连续值概率（0.0-1.0），表示AU的激活强度/置信度
  - `au_act`：二值激活标记（0或1），表示AU是否显著激活

- **时间分辨率**：帧级别标注，与30 FPS视频时间对齐

- **序列统计**：
  - 训练集平均长度：2071帧（约69秒）
  - 验证集平均长度：1780帧（约59秒）

### 数据模态

| 模态 | 格式 | 来源 | 用途 |
|------|------|------|------|
| 视频 | MP4, 30 FPS | Speaker + Listener | 面部表情、头部运动、视觉韵律 |
| 音频 | WAV | Speaker | 语音内容、韵律、音调、情感 |
| AU标注 | CSV | Listener | 训练目标和评估标准 |

---

## 技术挑战

### 1. **AU激活的稀疏性与不平衡**

**问题描述**：
- **稀疏性**：在大部分时间段内，多数AU处于未激活状态（`au_act=0`），正样本稀疏
- **变化差异**：不同AU的激活频率和强度差异显著
  - 高频AU（如AU12-微笑）：变化明显，易于学习
  - 低频AU（如AU9-鼻子皱起）：出现次数少，数据不足
  - 微弱AU（如AU18-嘴唇撅起）：即使激活也强度较低，难以检测

**算法设计考虑**：
- 使用**Focal Loss**和**加权损失函数**，为低频AU（如AU9）分配更高权重
- 在Qwen2.5 Omni的AU预测头中，为每个AU使用独立的输出层
- 训练时采用**分层采样**：过采样包含低频AU的样本
- 评估时**分AU报告性能**，避免被高频AU主导
- 可引入**对比学习**：拉近相似AU的表征，推开不同AU

### 2. **实时反应生成速度**

**问题描述**：
- 在真实对话场景中，听众的反应需要**低延迟**（理想<100ms，可接受<300ms）
- 模型推理需要在每帧或每几帧内完成（30 FPS → 33ms/frame）
- 需要在线处理音视频流，无法等待完整序列

**算法设计考虑**：
- **Qwen2.5 Omni的流式推理**：
  - 使用**滑动窗口**（1-2秒上下文）进行因果推理
  - 启用**KV-cache**复用，避免重复计算历史帧
  - **增量编码**：仅编码新增帧，复用已编码特征
  
- **模型轻量化**：
  - 使用**LoRA微调**而非全参数，减少显存和计算
  - **知识蒸馏**：大模型（Qwen2.5-32B）→小模型（Qwen2.5-7B/1.5B）
  - **INT8量化**：推理时使用量化模型，速度提升2-4倍
  
- **预测策略**：
  - **因果注意力**：只看历史和当前帧，不看未来
  - **提前预测**：预测未来3-5帧（100-166ms），为后续处理争取时间
  - **多尺度处理**：对关键帧进行高质量推理，中间帧用插值

### 3. **机器人平台的物理约束与延迟**

**问题描述**：
- **机械延迟**：机器人面部机械结构从指令到完成动作存在物理延迟（通常50-200ms）
- **网络延迟**：如果模型部署在云端，网络传输会引入额外延迟（10-100ms）
- **动作平滑性**：机械运动需要平滑过渡，突变会显得不自然或超出机械能力

**算法设计考虑**：
- **训练时建模延迟**：
  - 数据标注时加入**时间偏移**：输入t时刻，目标t+τ时刻（τ=100-200ms）
  - 让Qwen2.5 Omni学习"提前预测"，直接输出未来时刻的AU
  
- **后处理平滑**：
  - **高斯滤波**或**卡尔曼滤波**平滑AU时序
  - **速度限制**：`|AU[t] - AU[t-1]| < max_delta`，符合机械约束
  - **加速度约束**：限制二阶导数，避免突变
  
- **分层部署**：
  - **云端/GPU服务器**：运行Qwen2.5 Omni，生成AU序列
  - **机器人本地**：接收AU指令，映射到电机，执行运动
  - **缓冲队列**：维持200-500ms的AU预测缓冲，应对网络波动
  
- **自适应延迟补偿**：
  - 实时测量**端到端延迟**（输入到机器人动作完成）
  - 动态调整预测时间窗：`τ = measured_delay + safety_margin`
  - 如果延迟过大（>500ms），降低帧率但保持关键表情（如微笑、点头）

---

## 技术方案：基于Qwen2.5 Omni的端到端架构

### 核心模型选择

**Qwen2.5 Omni** - 多模态大语言模型

**选择理由**：
- **原生多模态输入**：直接处理视频+音频，无需手动特征提取
- **端到端学习**：从原始信号到AU预测的统一优化
- **强大的时序理解**：基于Transformer的长序列建模能力
- **预训练优势**：在大规模多模态数据上预训练，具备丰富的视听理解能力
- **灵活性**：可通过提示工程或微调适配特定任务

### 模型架构设计

#### 整体流程

```
输入：Speaker视频（MP4） + Speaker音频（WAV）
                    ↓
        [Qwen2.5 Omni 编码器]
          - 视频编码（直接处理帧序列）
          - 音频编码（直接处理波形/频谱）
          - 跨模态融合（内置attention机制）
                    ↓
            [任务适配层]
          - AU预测头（17个AU）
          - 时序解码器（生成帧级序列）
                    ↓
输出：Listener AU序列（17维 × T帧）
    - au_prob: 连续值激活概率
    - au_act: 二值激活标记
```

#### 关键模块

**1. 输入预处理**：
- **视频**：
  - 直接输入原始视频帧（或适当降采样，如15 FPS）
  - 帧尺寸调整至Qwen2.5 Omni要求（如224×224或更高）
  - 归一化处理
  
- **音频**：
  - 直接输入音频波形或mel-spectrogram
  - 采样率对齐（如16kHz）
  - 与视频时间对齐

**2. Qwen2.5 Omni微调策略**：
- **LoRA微调**：冻结主干，仅训练低秩适配器，降低计算成本
- **全参数微调**：如果资源充足，可获得更好性能
- **提示工程**：设计任务提示，引导模型理解任务
  ```
  示例提示：
  "分析说话者的视频和音频，预测听众的面部动作单元（AU）激活序列。
  输出17个AU在每一帧的激活概率。"
  ```

**3. 任务适配头**：
- **回归头**：预测`au_prob`（17维向量，值域0-1）
  - 使用Sigmoid激活
  - 独立预测每个AU（处理AU间不平衡）
  
- **分类头**：预测`au_act`（17维二值向量）
  - 使用阈值或独立的分类器
  - 可与回归头共享特征

- **时序对齐**：
  - 输出长度与输入帧数对齐
  - 使用时序卷积或线性插值

**4. 延迟补偿模块**（针对挑战3）：
- 在解码器中加入**未来预测机制**
- 模型预测未来τ时刻的AU（τ=100-200ms）
- 训练时输入当前帧t，目标为t+τ时刻的AU

### 训练策略

#### 损失函数设计

针对AU稀疏性和不平衡问题（挑战1）：

```python
# 多任务损失
total_loss = λ1 * regression_loss + λ2 * classification_loss + λ3 * temporal_loss

# 1. 回归损失（au_prob预测）
# 使用加权MSE，为低频AU赋予更高权重
regression_loss = Σ(w_i * MSE(pred_prob_i, true_prob_i))

# 2. 分类损失（au_act预测）
# 使用Focal Loss处理类别不平衡
classification_loss = Σ(-α * (1-p)^γ * log(p))

# 3. 时序一致性损失
# 约束相邻帧的AU变化平滑（针对挑战3的平滑性）
temporal_loss = Σ||pred[t] - pred[t-1]||^2

# AU权重计算（基于激活频率）
w_i = 1 / log(activation_frequency_i + ε)
```

#### 数据增强

- **视频增强**：
  - 随机裁剪、翻转（水平）
  - 颜色抖动、亮度/对比度调整
  - 时间采样（随机选择起始位置）
  
- **音频增强**：
  - 时间拉伸/压缩（±10%）
  - 音量调整、添加背景噪声
  - SpecAugment（频率/时间遮蔽）
  
- **时序增强**：
  - 随机裁剪固定长度片段（如5-10秒）
  - 滑动窗口训练

#### 实时性优化（针对挑战2）

**推理加速策略**：

1. **模型量化**：
   - INT8/FP16量化，减少计算量
   - 使用量化感知训练（QAT）保持精度

2. **流式推理**：
   - 使用滑动窗口（如1秒窗口，30帧）
   - KV-cache优化，复用已计算的attention
   - 增量式推理，仅处理新增帧

3. **模型压缩**：
   - 知识蒸馏：用小模型（如Qwen2.5-1.5B）学习大模型
   - 剪枝：移除不重要的attention头和FFN层

4. **并行处理**：
   - 视频和音频编码并行
   - 使用TensorRT/ONNX Runtime加速

5. **预测策略**：
   - 因果推理：仅使用历史信息
   - 固定延迟：预测未来100-200ms（补偿机械延迟）

**性能目标**：
- 推理延迟：<50ms（模型推理）
- 端到端延迟：<200ms（包含预处理+推理+后处理）
- 吞吐量：>30 FPS

#### 机器人部署优化（针对挑战3）

**分层架构**：

```
[云端/边缘服务器]
    ├── Qwen2.5 Omni模型（主推理）
    ├── AU序列生成
    ├── 延迟预测（提前τ时间）
    └── 平滑滤波（高斯/卡尔曼）
            ↓ (网络传输)
[机器人本地控制器]
    ├── AU到电机映射
    ├── 运动插值（保证平滑）
    ├── 缓冲队列（应对网络抖动）
    └── 实时反馈调整
```

**延迟补偿训练**：
```python
# 训练时显式建模延迟
# 输入：t时刻的speaker视频/音频
# 目标：t+delay时刻的listener AU
delay = 100  # ms，约3帧（30 FPS）
target_frame = current_frame + delay // (1000 / fps)
```

**后处理平滑**：
```python
# 高斯滤波平滑AU序列
au_smoothed = gaussian_filter1d(au_predicted, sigma=2, axis=0)

# 限制变化率（符合机械约束）
au_clipped = clip_velocity(au_smoothed, max_delta=0.1)
```

### 评估指标

**AU预测性能**：
- **分AU指标**：
  - F1分数、精确率、召回率（针对`au_act`）
  - MAE、RMSE、Pearson相关系数（针对`au_prob`）
  - 按AU激活频率分组评估（高频/中频/低频）
  
- **序列级指标**：
  - 动态时间规整（DTW）距离
  - 时序相关性（Cross-correlation）

**实时性能**：
- 推理时间（ms/frame）
- 吞吐量（FPS）
- 端到端延迟（输入→输出）

**机器人适配性**：
- AU变化率统计（确保在机械限制内）
- 平滑度评分（相邻帧差异）
- 延迟补偿精度（预测vs实际的时间偏移）

---

## 数据加载示例

```python
from datasets import load_from_disk
import cv2
import librosa

# 加载数据集
train_dataset = load_from_disk('/net/scratch/k09562zs/LLM_reaction_Robot/Reaction_DataSet/processed/train')
val_dataset = load_from_disk('/net/scratch/k09562zs/LLM_reaction_Robot/Reaction_DataSet/processed/val')

# 访问样本
sample = train_dataset[0]

# 加载说话者视频（输入）
cap = cv2.VideoCapture(sample['speaker_video_path'])
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# 加载说话者音频（输入）
audio, sr = librosa.load(sample['speaker_audio_path'], sr=16000)

# 获取听众AU标注（目标）
listener_au_prob = sample['listener_au_prob']  # Dict[str, List[float]]
listener_au_act = sample['listener_au_act']    # Dict[str, List[int]]

# AU17的概率序列
au17_probs = listener_au_prob['AU17']  # List[float], length = n_frames
```

---

## 项目文件结构

```
LLM_reaction_Robot/
├── README.md                          # 本文档
├── Tools/
│   ├── prepare_dataset.py             # 数据集预处理脚本
│   └── test_data_loading.ipynb        # 数据加载测试notebook
├── Reaction_DataSet/
│   ├── train/                         # 原始训练数据
│   ├── val/                           # 原始验证数据
│   └── processed/
│       ├── train/                     # 处理后的训练集（1,660样本）
│       └── val/                       # 处理后的验证集（571样本）
└── (待开发)
    ├── models/                        # 模型定义
    ├── training/                      # 训练脚本
    └── inference/                     # 推理与部署
```

---

## 下一步工作

1. **环境准备与模型部署**：
   - 安装Qwen2.5 Omni及其依赖（transformers, torch等）
   - 下载预训练模型权重
   - 测试基本的视频+音频输入能力

2. **数据适配**：
   - 将数据集格式转换为Qwen2.5 Omni的输入格式
   - 设计任务提示模板
   - 实现数据加载器（支持批处理和流式）

3. **基线模型训练**：
   - 实现LoRA微调pipeline
   - 训练基础AU预测模型（不考虑延迟）
   - 评估在验证集上的AU预测性能

4. **实时性优化**：
   - 实现流式推理（滑动窗口+KV-cache）
   - 测试推理速度和延迟
   - 应用模型量化和蒸馏

5. **延迟补偿与机器人集成**：
   - 加入时间偏移训练（预测未来τ时刻）
   - 开发AU到机器人电机的映射接口
   - 端到端测试：视频输入→AU预测→机器人动作

6. **数据分析**（辅助优化）：
   - 统计每个AU的激活频率分布
   - 分析说话者-听众的时间延迟模式
   - 可视化高频vs低频AU的特征

---

## 参考资源

- **数据准备**：`Tools/prepare_dataset.py`
- **数据加载测试**：`Tools/test_data_loading.ipynb`
- **Qwen2.5 Omni官方文档**：https://github.com/QwenLM/Qwen2.5
- **HuggingFace Datasets**：https://huggingface.co/docs/datasets
- **LoRA微调**：https://github.com/microsoft/LoRA

---

**最后更新**：2026年1月2日
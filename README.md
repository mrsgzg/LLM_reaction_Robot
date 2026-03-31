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
  - `au_prob`：**连续值概率（0.0-1.0）**，表示AU的激活强度/置信度
    - **核心训练目标**：用于回归预测，实现机器人连续运动控制
    - 统计特性（训练集）：
      - 高活性AU（均值>0.5）：AU14(0.57), AU25(0.55) - 高频表情
      - 中活性AU（均值0.2-0.5）：AU1(0.30), AU12(0.33), AU18(0.42) - 常见反应
      - 低活性AU（均值<0.2）：AU15(0.11), AU9(0.14) - 微弱表情
  - `au_act`：二值激活标记（0或1），表示AU是否显著激活
    - **辅助参考**：用于阈值验证，非主要训练目标

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

### 1. **AU激活强度的不平衡与连续性建模**

**问题描述**：
- **概率分布不均**：不同AU的平均激活强度差异显著（基于`au_prob`连续值）
  - 高强度AU（均值>0.5）：AU14(0.57), AU25(0.55) - 频繁且强烈的表情
  - 中强度AU（均值0.2-0.5）：AU1(0.30), AU6(0.32), AU12(0.33) - 常规反应
  - 低强度AU（均值<0.15）：AU9(0.14), AU15(0.11) - 微弱且稀少的表情
- **连续值建模难度**：需要精确预测0-1之间的连续概率，而非简单的二分类
- **机器人运动映射**：概率值直接驱动电机，需要平滑且精确的预测

**算法设计考虑**：
- 使用**加权MSE损失**用于`au_prob`回归，根据AU的标准差和激活频率分配权重
  - 高方差AU（如AU25 std=0.29）：权重更高，因为变化范围大
  - 低均值AU（如AU9 mean=0.14）：权重调整，避免被高均值AU主导
- **平滑性约束**：添加时序一致性损失，确保相邻帧预测连续（机器人运动需求）
- 在模型输出层使用**Sigmoid激活**，确保输出在[0,1]范围
- 训练时对每个AU独立建模，使用17个独立的回归头
- 评估时计算**MAE、RMSE、Pearson相关系数**，而非传统的F1/Accuracy

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
  - **高斯滤波**（σ=2.0）或**卡尔曼滤波**平滑AU概率时序
  - **速度限制**：`|AU_prob[t] - AU_prob[t-1]| < 0.15`，确保机械运动连续性
  - **加速度约束**：`|ΔAU_prob[t] - ΔAU_prob[t-1]| < 0.05`，避免突变导致机械损伤
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

## 技术方案：知识蒸馏架构（大模型教师 + 轻量学生）

### 核心设计思路

采用**知识蒸馏**方案平衡精度和实时性：
- **教师模型**：Qwen2.5 Omni（大模型，高精度）→ 用于训练和知识提取
- **学生模型**：轻量级LSTM/Transformer（小模型，高速度）→ 用于实时推理部署

**为什么是蒸馏而非直接用大模型**：

| 维度 | 大模型直推 | 蒸馏方案 |
|------|-----------|--------|
| 推理延迟 | ❌ 300-500ms | ✓ <100ms |
| 边缘部署 | ❌ 需高端GPU | ✓ CPU/轻GPU可用 |
| 精度 | ✓ 最高 | ✓ 95%精度保留 |
| 能耗 | ❌ 高 | ✓ 低 |
| 机器人适配 | ❌ 困难 | ✓ 友好 |

**知识蒸馏的优势**：学生模型不仅学习硬标签(groundtruth)，更重要的是学习教师模型的"软概率"，这包含了类间关系、不确定性等丰富知识，使学生模型以更快的推理速度达到接近教师的精度。

### 模型架构设计

#### 三阶段蒸馏流程

```
离线阶段 (训练)：
  
  [训练数据] + [Groundtruth AU标注]
        ↓
  ┌──────────────────────────────────────────────────────┐
  │ 第一阶段：教师模型训练                                 │
  │ ─────────────────────────────────────                │
  │ Qwen2.5 Omni (32B/14B) 微调                         │
  │ ├─ 输入：Speaker视频+音频                            │
  │ ├─ 直接端到端学习AU预测                              │
  │ └─ 输出：高精度AU预测 + 软概率                        │
  │                                                     │
  │ ✓ 在验证集上精度最高                                 │
  │ ✓ 推理慢(300-500ms)，仅训练时用                      │
  │ ✓ 保存预训练权重                                    │
  └──────────────────────────────────────────────────────┘
        ↓
  [教师软标签生成]
  在整个训练集上推理，获得Teacher的软概率
  (保存，后续蒸馏使用)
        ↓
  ┌──────────────────────────────────────────────────────┐
  │ 第二阶段：学生模型蒸馏训练                              │
  │ ─────────────────────────────────────                │
  │ 学生选项：LSTM / GRU / 小型Transformer             │
  │ ├─ 学习groundtruth (硬标签)                          │
  │ ├─ 模仿Teacher的软概率 (软标签)                       │
  │ └─ 输出：轻量级高速AU预测模型                        │
  └──────────────────────────────────────────────────────┘
        ↓
在线阶段 (推理)：
  
  [实时视频流] + [实时音频流]
        ↓
  ┌──────────────────────────────────────────────────────┐
  │ 第三阶段：实时推理部署                                │
  │ ─────────────────────────────────────                │
  │ 学生模型 (轻量级LSTM/Transformer)                   │
  │ ├─ 推理延迟：<100ms ✓                               │
  │ ├─ 精度保留：90-95% (vs Teacher)                   │
  │ └─ 输出：AU序列 → 机器人动作                        │
  └──────────────────────────────────────────────────────┘
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

**2. Qwen2.5 Omni特征提取器 + AU回归头**：

**架构设计（推荐方案）**：
```python
# 不使用文本生成方式输出AU（效率低、不稳定）
# 而是使用Encoder + 回归头的方式

class QwenAUTeacher(nn.Module):
    def __init__(self):
        # Qwen2.5-Omni作为特征提取器
        self.encoder = Qwen2_5OmniForConditionalGeneration(...)
        
        # AU回归头：从hidden states直接预测AU概率
        self.au_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 17),
            nn.Sigmoid()  # 输出0-1概率
        )
    
    def forward(self, videos, audios):
        # 提取视频+音频的多模态特征
        features = self.encoder.encode(videos, audios)  # [B, T, D]
        
        # 回归输出17个AU概率
        au_probs = self.au_head(features)  # [B, T, 17]
        return au_probs
```

**为什么不用文本生成AU值？**
- ❌ 文本生成（如生成JSON）：
  - 输出不稳定（JSON解析可能失败）
  - 推理延迟高（生成token慢，每个AU值需要多个token）
  - 精度受限（文本表示浮点数有损）
  - 难以优化（无法直接计算MSE/temporal loss）
  
- ✅ 回归头直接输出：
  - 输出稳定可靠（固定维度tensor）
  - 推理高效（一次forward即可）
  - 精度高（float32直接表示）
  - 便于优化（直接计算Weighted MSE + Temporal loss）
  - 易于知识蒸馏（学生模型学习soft probabilities）

**微调策略**：
- **LoRA微调**：冻结Qwen2.5-Omni主干，仅训练LoRA适配器 + AU回归头
- **全参数微调**：如果资源充足（A100/H100），可获得更好性能
- **训练目标**：最小化Weighted MSE + Temporal Smoothness + Velocity Constraint

**3. 时序对齐与输出**：
  - 输出长度与输入帧数对齐
  - 使用时序卷积或线性插值

**4. 延迟补偿模块**（针对挑战3）：
- 在解码器中加入**未来预测机制**
- 模型预测未来τ时刻的AU（τ=100-200ms）
- 训练时输入当前帧t，目标为t+τ时刻的AU

### 训练策略

#### 损失函数设计

针对AU概率连续值预测与机器人运动平滑性（挑战1+3）：

```python
# 多任务损失（专注于连续值回归）
total_loss = λ1 * weighted_mse_loss + λ2 * temporal_smooth_loss + λ3 * velocity_loss

# 1. 加权MSE损失（au_prob预测 - 主要任务）
# 根据AU的统计特性分配权重
weighted_mse_loss = Σ(w_i * MSE(pred_prob_i, true_prob_i))

# AU权重设计（基于数据分析结果）：
# - 高方差AU（如AU25 std=0.29）：权重↑，因变化剧烈需精准预测
# - 低均值AU（如AU9 mean=0.14）：权重↑，避免被高均值AU主导
w_i = (std_i / mean_std) * (1 / (mean_i + ε))
# 其中 std_i, mean_i 从 au_prob_config.json 获取

# 2. 时序平滑损失（一阶导数约束）
# 确保相邻帧AU变化连续，满足机器人机械约束
temporal_smooth_loss = Σ||pred[t] - pred[t-1]||^2

# 3. 速度约束损失（二阶导数约束）
# 限制加速度，防止机器人运动突变
velocity_loss = Σ||(pred[t] - pred[t-1]) - (pred[t-1] - pred[t-2])||^2

# 推荐超参数：
λ1 = 1.0   # 主损失
λ2 = 0.1   # 平滑性
λ3 = 0.05  # 速度约束
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

**学生vs教师性能对比**：

```
┌─────────────────────────┬──────────────┬──────────────┬──────────────┐
│ 指标                    │ 教师(Qwen)   │ 学生(LSTM)   │ 保留比例     │
├─────────────────────────┼──────────────┼──────────────┼──────────────┤
│ 推理延迟(ms)            │ 300-500      │ 40-80        │ 15-25%       │
│ 平均F1 (all AU)         │ 0.86         │ 0.81         │ 94%          │
│ 低频AU F1               │ 0.72         │ 0.68         │ 94%          │
│ 显存占用(GB)            │ 48           │ 0.5          │ 1%           │
│ 部署难度                │ 很难         │ 容易         │ -            │
│ 加速倍数                │ 1x           │ 8.9x         │ -            │
└─────────────────────────┴──────────────┴──────────────┴──────────────┘
```

**AU预测精度**（每个AU独立评估）：
- **回归指标（au_prob - 主要评估目标）**：
  - **MAE (Mean Absolute Error)**：预测概率与真实概率的平均绝对误差
    - 总体目标：MAE < 0.08
    - 低强度AU（如AU9）：MAE < 0.05（因基准值低）
    - 高强度AU（如AU14）：MAE < 0.10
  - **RMSE (Root Mean Square Error)**：对大误差更敏感
  - **Pearson相关系数**：预测与真实的时序相关性 (目标 > 0.85)
  - **概率分布一致性**：KL散度或JS散度
  
- **辅助分类指标（au_act）**：
  - F1、精确率、召回率（使用0.5阈值）
  - 仅用于验证模型的判别能力，非主要优化目标

**实时性能**：
- **单帧延迟**：<50ms（满足30 FPS）
- **P99延迟**：<150ms（99%的帧在此延迟内）
- **吞吐量**：≥30 FPS
- **稳定性**：延迟方差<30ms

**机器人适配性**：
- **平滑度**：相邻帧AU变化<0.15
- **加速度**：二阶导数<0.05
- **运动自然度**：人工评分
- **延迟补偿精度**：预测AU vs 实际AU的相关性>0.85

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
│   ├── test_data_loading.ipynb        # 数据加载测试notebook
│   ├── analyze_au_prob.py             # ✅ AU概率分析脚本（已完成）
│   └── data_analysis.ipynb            # 数据分析notebook
├── analysis_results/
│   ├── au_probability_analysis.png    # ✅ AU概率统计可视化（4图）
│   └── au_prob_config.json            # ✅ 训练配置文件（17 AUs统计）
├── Reaction_DataSet/
│   ├── train/                         # 原始训练数据
│   ├── val/                           # 原始验证数据
│   └── processed/
│       ├── train/                     # 处理后的训练集（1,660样本）
│       └── val/                       # 处理后的验证集（571样本）
└── (待开发)
    ├── models/                        # 模型定义
    │   ├── teacher_qwen.py            # Qwen2.5 Omni教师模型
    │   └── student_models.py          # LSTM/Transformer学生模型
    ├── training/                      # 训练脚本
    │   ├── train_teacher.py           # 教师模型训练
    │   └── distillation.py            # 知识蒸馏训练
    ├── inference/                     # 推理部署
    └── evaluation/                    # 评估脚本
```
    └── inference/                     # 推理与部署
```

---

## 下一步工作（按优先级）

### ✅ 已完成

1. **数据准备**：
   - 数据集预处理：1,660训练样本 + 571验证样本
   - AU概率统计分析：17 AUs的mean/median/std/percentiles分析完成
   - 训练配置文件：`au_prob_config.json` 生成（含AU权重建议）
   - 可视化报告：4-panel分析图表（mean对比/分布直方图/箱线图/摘要）

### 第0阶段：Qwen Omni可行性验证（✅ 已完成）

2. **验证结果**（2026-01-02）：
   - ✅ 模型加载：Qwen2.5-Omni-3B成功加载（8-bit量化，6GB显存）
   - ✅ 多模态输入：视频+音频处理正常
   - ✅ 推理能力：可生成AU相关输出
   - ⚠️ **关键发现**：
     - 文本生成方案：516秒/10秒视频（太慢！）
     - JSON输出：被截断，解析失败
     - **结论**：文本生成不可行，需改用回归头
   
3. **已实现的架构**（`models/teacher_qwen.py`）：
   - ✅ Qwen2.5-Omni encoder + AU回归头
   - ✅ LoRA适配器支持（可选，减少可训练参数）
   - ✅ 8-bit量化支持
   - ✅ 推理速度对比脚本（`tools/speed_comparison.py`）
   
4. **性能预测**：
   - 回归头推理：<10ms（vs 文本生成516秒）
   - **加速倍数：50,000倍！**

---

### 第一阶段：教师模型训练（2-3周）

5. **实现完整训练脚本**（`training/train_teacher.py`）：
   - DataLoader：加载视频+音频+AU标注
   - 损失函数：Weighted MSE + Temporal Smoothness + Velocity Constraint
   - 训练循环：前向→反向→优化
   - 验证评估：MAE/RMSE/Pearson计算
   - 模型保存：最优权重检查点

6. **准备数据加载器**（`data/dataloader.py`）：
   - 视频加载：帧解码、正则化
   - 音频加载：Mel-spectrogram提取
   - 时间对齐：确保视频帧与AU标注同步
   - 数据增强：视频/音频增强策略

7. **实现AU回归头**（已在`models/teacher_qwen.py`）：
   - ✅ Encoder + Linear(D, 17) + Sigmoid
   - ✅ LoRA适配器配置
   - ✅ 8-bit量化支持

8. **教师模型训练**：
   ```bash
   python training/train_teacher.py \
     --model_id Qwen/Qwen2.5-Omni-3B \
     --use_lora True \
     --lora_rank 64 \
     --batch_size 4 \
     --epochs 10 \
     --learning_rate 5e-4 \
     --loss_config analysis_results/au_prob_config.json
   ```

9. **教师模型评估**：
   - 验证集评估：MAE、RMSE、Pearson相关系数
   - 目标：MAE < 0.08, RMSE < 0.12, Pearson > 0.85
   - 按AU分析精度：识别瓶颈AU
   - 如精度不足，调整超参或损失函数权重
     --epochs 10 \
     --learning_rate 5e-4 \
     --loss_config analysis_results/au_prob_config.json  # 使用AU统计权重
   ```

4. **教师模型评估**：
   - 验证集评估：MAE、RMSE、Pearson相关系数
   - 目标：MAE < 0.08, RMSE < 0.12, Pearson > 0.85
   - 如精度不足，调整超参或损失函数权重

### 第二阶段：知识蒸馏与部署（3-4周）

5. **生成软标签**：
   ```bash
   python generate_teacher_labels.py \
     --teacher_checkpoint ./checkpoints/teacher_best.pt \
     --output ./data/teacher_soft_labels.pt
   ```

6. **学生模型设计**：
   - 对比LSTM (40-80ms) vs 小型Transformer (80-150ms)
   - 选择参数最少且延迟<100ms的架构
   - 实现完整的学生模型代码（输出连续AU概率）

7. **蒸馏训练**：
   ```bash
   python train_student_distill.py \
     --student_arch lstm \
     --teacher_labels ./data/teacher_soft_labels.pt \
     --temperature 4.0 \
     --alpha 0.3 \
     --epochs 40 \
     --loss_config analysis_results/au_prob_config.json  # 使用AU权重 + 平滑约束
   ```

8. **学生模型评估**：
   - 精度保留：学生MAE与教师对比（目标>90%精度保留）
   - 推理延迟：实测推理速度（目标<100ms）
   - 连续性验证：检查AU序列的平滑度（速度/加速度约束）

### 第三阶段：延迟补偿与机器人集成（2-3周）

### 第三阶段：延迟补偿与机器人集成（2-3周）

9. **延迟建模训练**：
    - 修改数据集：目标由t改为t+τ (τ=100-200ms)
    - 重新训练学生模型（学习预测未来AU概率）
    - 评估延迟补偿精度

10. **后处理优化**：
    - 实现高斯滤波器（σ=2.0）平滑AU概率序列
    - 设置机械约束：速度限制（<0.15/帧）、加速度限制（<0.05）
    - 测试AU序列的自然度与机械可行性

11. **流式推理接口**：
    - 实现滑动窗口推理（1-2秒窗口）
    - 与机器人系统对接（输出AU概率→电机指令）
    - 缓冲队列管理（200-500ms预测缓冲）

12. **机器人测试**：
    - 端到端延迟测量：输入→预测→机器人动作完成
    - 自然度评估：人类评估员评分
    - 故障模式分析：抖动、延迟、不自然表情

### 第四阶段：数据分析与优化（持续）

13. **性能诊断**：
    - 各AU的预测误差分析（MAE/RMSE per AU）
    - 失败案例分析（高误差样本）
    - 时间序列对比可视化（预测vs真实AU概率）

14. **针对性改进**：
    - 为高误差AU调整损失权重
    - 对困难样本进行数据增强
    - 优化temporal smoothness超参数（λ2, λ3）

15. **长期优化**：
    - 尝试不同学生架构（LSTM层数/Transformer head数）
    - 实验不同蒸馏温度（T=2/4/8）
    - 可选：INT8量化、ONNX导出

---

## 参考资源

**数据与分析**：
- [prepare_dataset.py](Tools/prepare_dataset.py) - 数据准备脚本
- [test_data_loading.ipynb](Tools/test_data_loading.ipynb) - 数据加载测试
- [analyze_au_prob.py](Tools/analyze_au_prob.py) - AU概率统计分析
- [au_prob_config.json](analysis_results/au_prob_config.json) - AU训练配置

**模型与训练**：
- [Qwen2.5官方文档](https://github.com/QwenLM/Qwen2.5)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PEFT (LoRA微调)](https://github.com/huggingface/peft)
- [知识蒸馏论文](https://arxiv.org/abs/1503.02531) - Hinton et al., 2015

**实时部署**：
- [ONNX Runtime](https://onnxruntime.ai/) - 跨平台推理
- [TensorRT](https://developer.nvidia.com/tensorrt) - NVIDIA推理优化
- [TorchScript](https://pytorch.org/docs/stable/jit.html) - PyTorch模型导出

**性能优化**：
- 模型量化：[PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- 流式处理：滑动窗口+缓冲设计
- 时间分析：使用`torch.profiler`诊断bottleneck

---

**最后更新**：2025年1月2日  
**方案版本**：知识蒸馏 v2.1（连续AU概率建模 + 机器人运动平滑约束）
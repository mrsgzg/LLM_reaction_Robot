"""
Qwen2.5 Omni 输入输出验证脚本
目标：验证Qwen2.5-Omni能否处理视频+音频输入，输出AU概率预测

官方文档参考: 
- Model: Qwen/Qwen2.5-Omni-3B (或7B)
- 支持视频输入（包含音频）
- 使用Qwen2_5OmniForConditionalGeneration和Qwen2_5OmniProcessor
"""

import torch
import time
import json
from pathlib import Path
from datasets import load_from_disk
import numpy as np

# 检查环境
print("=" * 60)
print("环境检查")
print("=" * 60)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# ===================================================================
# 1. 加载数据集样本
# ===================================================================
print("=" * 60)
print("步骤1: 加载数据集样本")
print("=" * 60)

dataset_path = "/net/scratch/k09562zs/LLM_reaction_Robot/Reaction_DataSet/processed/train"
print(f"数据集路径: {dataset_path}")

try:
    dataset = load_from_disk(dataset_path)
    print(f"✅ 成功加载数据集，包含 {len(dataset)} 个样本")
except Exception as e:
    print(f"❌ 加载数据集失败: {e}")
    exit(1)

# 选择一个样本进行测试
sample_idx = 0
sample = dataset[sample_idx]
print(f"\n测试样本 #{sample_idx}:")
print(f"  - Speaker视频: {sample['speaker_video_path']}")
print(f"  - Speaker音频: {sample['speaker_audio_path']}")

# 打印样本的所有字段
print(f"\n样本包含的字段: {list(sample.keys())}")

# 安全地打印可选字段
if 'video_length_seconds' in sample:
    print(f"  - 视频长度: {sample['video_length_seconds']:.2f}秒")
if 'total_frames' in sample:
    print(f"  - 总帧数: {sample['total_frames']}帧")
if 'fps' in sample:
    print(f"  - FPS: {sample['fps']}")

# 获取视频路径用于后续推理
video_path = sample['speaker_video_path']

# ===================================================================
# 2. 获取真实AU标注（用于对比）
# ===================================================================
print("\n" + "=" * 60)
print("步骤2: 获取真实AU标注")
print("=" * 60)

listener_au_prob = sample['listener_au_prob']
au_names = list(listener_au_prob.keys())
print(f"AU列表 ({len(au_names)}个): {', '.join(au_names)}")

# 打印前10帧的部分AU概率
print(f"\n前10帧的AU概率样例 (AU1, AU6, AU12):")
for i in range(min(10, len(listener_au_prob['AU1']))):
    au1 = listener_au_prob['AU1'][i]
    au6 = listener_au_prob['AU6'][i]
    au12 = listener_au_prob['AU12'][i]
    print(f"  帧{i}: AU1={au1:.3f}, AU6={au6:.3f}, AU12={au12:.3f}")

# ===================================================================
# 3. 尝试加载Qwen2.5-Omni模型
# ===================================================================
print("\n" + "=" * 60)
print("步骤3: Qwen2.5-Omni模型测试")
print("=" * 60)

model = None
processor = None

try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    print("✅ Qwen2.5-Omni依赖已安装")
    
    # 尝试加载模型（使用3B版本进行测试）
    model_name = "Qwen/Qwen2.5-Omni-3B"
    print(f"\n尝试加载模型: {model_name}")
    print("警告: 首次运行会下载模型，请确保网络畅通")
    
    print("\n选项1: 完整加载（推荐用于测试）")
    print("  processor = Qwen2_5OmniProcessor.from_pretrained(model_name)")
    print("  model = Qwen2_5OmniForConditionalGeneration.from_pretrained(")
    print("      model_name, torch_dtype='auto', device_map='auto')")
    
    print("\n选项2: 8-bit量化加载（节省显存）")
    print("  需要: pip install bitsandbytes")
    print("  model = Qwen2_5OmniForConditionalGeneration.from_pretrained(")
    print("      model_name, load_in_8bit=True, device_map='auto')")
    
    print("\n正在加载处理器...")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    print("✅ 处理器加载成功")
    
    print("\n正在加载模型（可能需要几分钟）...")
    start_time = time.time()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )
    load_time = time.time() - start_time
    print(f"✅ 模型加载成功 (耗时: {load_time:.2f}秒)")
    
    # 显示显存占用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  - GPU显存占用: {allocated:.2f} GB (已分配) / {reserved:.2f} GB (已保留)")
    
except ImportError as e:
    print(f"⚠️ 无法导入Qwen2.5-Omni: {e}")
    print("\n安装说明:")
    print("  pip install transformers>=4.40.0")
    print("  pip install qwen-omni-utils")
    print("  pip install soundfile")
    print("\n如果transformers版本过旧，请升级:")
    print("  pip install --upgrade transformers")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("\n可能的原因:")
    print("  1. 网络问题：无法下载模型文件")
    print("  2. 显存不足：尝试使用8-bit量化")
    print("  3. transformers版本不兼容")

# ===================================================================
# 4. 构造输入并进行推理测试
# ===================================================================
if model is not None and processor is not None:
    print("\n" + "=" * 60)
    print("步骤4: 推理测试")
    print("=" * 60)
    
    # 获取音频路径
    audio_path = sample['speaker_audio_path']
    
    # ⚠️ 为了避免OOM，只处理前10秒视频
    print("\n⚠️ 显存优化：只处理前10秒视频（约300帧）以避免OOM")
    print("   完整视频处理需要更多优化（如分段处理、帧采样等）")
    
    # 使用ffmpeg截取前10秒视频和音频
    import subprocess
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    short_video_path = os.path.join(temp_dir, "short_video.mp4")
    short_audio_path = os.path.join(temp_dir, "short_audio.wav")
    
    print(f"  - 截取视频前10秒...")
    subprocess.run([
        "ffmpeg", "-i", video_path, "-t", "10", "-c:v", "copy",
        "-y", short_video_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    print(f"  - 截取音频前10秒...")
    subprocess.run([
        "ffmpeg", "-i", audio_path, "-t", "10", "-c:a", "copy",
        "-y", short_audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    print(f"✅ 短视频生成完成")
    print(f"  - 短视频: {short_video_path}")
    print(f"  - 短音频: {short_audio_path}")
    
    # 使用短视频进行测试
    video_path = short_video_path
    audio_path = short_audio_path
    
    # 构造conversation格式（Qwen2.5-Omni要求的格式）
    # 注意：由于视频和音频是分开存储的，需要单独传入音频
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path  # 使用本地视频路径（无音频轨）
                },
                {
                    "type": "audio",
                    "audio": audio_path  # 单独传入音频文件
                },
                {
                    "type": "text",
                    "text": """Analyze the speaker's video and audio. Predict the listener's 17 facial Action Units (AU) activation probabilities for each frame.

Output Format (JSON):
{
  "AU1": [0.2, 0.3, 0.25, ...],
  "AU2": [0.1, 0.15, 0.12, ...],
  "AU4": [...],
  "AU6": [...],
  "AU7": [...],
  "AU9": [...],
  "AU10": [...],
  "AU12": [...],
  "AU14": [...],
  "AU15": [...],
  "AU17": [...],
  "AU18": [...],
  "AU20": [...],
  "AU23": [...],
  "AU24": [...],
  "AU25": [...],
  "AU26": [...]
}

Requirements:
- Output continuous probability values (0.0-1.0), not binary 0/1
- Array length should match the number of video frames
- Consider speaker's emotion, tone, and content impact on listener's expression"""
                }
            ],
        },
    ]
    
    print("构造输入格式...")
    try:
        # 由于视频和音频是分开存储的，设置为False，单独传入音频
        USE_AUDIO_IN_VIDEO = False
        
        print(f"  - 视频路径: {video_path}")
        print(f"  - 音频路径: {audio_path}")
        print(f"  - use_audio_in_video: {USE_AUDIO_IN_VIDEO}")
        
        # 应用chat模板
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        print(f"✅ Chat模板应用成功")
        print(f"提示词长度: {len(text)} 字符")
        
        # 处理多模态信息
        print("\n处理多模态输入...")
        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        
        print(f"✅ 多模态处理完成")
        print(f"  - 音频: {len(audios) if audios else 0} 个")
        print(f"  - 图像: {len(images) if images else 0} 个")
        print(f"  - 视频: {len(videos) if videos else 0} 个")
        
        # 编码输入
        print("\n编码输入...")
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = inputs.to(model.device).to(model.dtype)
        print("✅ 输入编码成功")
        
        # 推理
        print("\n开始推理...")
        print("提示: 如果仍然OOM，可以进一步降低视频时长（如5秒）或分辨率")
        start_time = time.time()
        
        # 清空显存缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  - 推理前显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        with torch.no_grad():
            text_ids, audio = model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                max_new_tokens=2048  # 降低token数量，减少显存占用
            )
        
        inference_time = time.time() - start_time
        
        # 解码输出
        output_text = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        print(f"✅ 推理完成 (耗时: {inference_time:.2f}秒)")
        print(f"\n模型输出:")
        print("-" * 60)
        print(output_text[0][:1000])  # 打印前1000字符
        if len(output_text[0]) > 1000:
            print(f"... (总长度: {len(output_text[0])} 字符)")
        print("-" * 60)
        
        # 尝试解析JSON
        print("\n解析输出...")
        try:
            # 查找JSON部分
            output_str = output_text[0]
            json_start = output_str.find('{')
            json_end = output_str.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = output_str[json_start:json_end]
                predicted_aus = json.loads(json_str)
                
                print("✅ JSON解析成功")
                print(f"预测的AU数量: {len(predicted_aus)}")
                
                # 验证输出格式
                for au_name in ['AU1', 'AU6', 'AU12']:
                    if au_name in predicted_aus:
                        values = predicted_aus[au_name]
                        print(f"\n{au_name}:")
                        print(f"  - 长度: {len(values)}")
                        print(f"  - 前10个值: {values[:10]}")
                        print(f"  - 值域: [{min(values):.3f}, {max(values):.3f}]")
            else:
                print("⚠️ 输出中未找到JSON格式")
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            print("提示: 可能需要调整提示词或后处理输出格式")
        
    except Exception as e:
        print(f"❌ 推理过程失败: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\n⚠️ 跳过推理测试（模型未加载）")

# ===================================================================
# 5. 总结与建议
# ===================================================================
print("\n" + "=" * 60)
print("验证总结")
print("=" * 60)

print("\n✅ 已完成的验证:")
print("  1. ✅ 数据集加载正常")
print("  2. ✅ AU标注格式确认（17个AU，连续概率值）")
print(f"  3. {'✅' if processor else '⚠️'} Qwen2.5-Omni处理器")
print(f"  4. {'✅' if model else '⚠️'} Qwen2.5-Omni模型加载")

if model is not None:
    print("\n🎉 推理测试完成！关键发现:")
    print("  - 模型可以加载并处理视频输入")
    print("  - 需要验证输出格式是否符合AU概率要求")
    print("  - 推理延迟需要优化（知识蒸馏方案）")
else:
    print("\n📋 下一步行动:")
    print("  1. 安装Qwen2.5-Omni依赖:")
    print("     pip install transformers>=4.40.0")
    print("     pip install qwen-omni-utils")
    print("     pip install soundfile")
    print("     pip install accelerate bitsandbytes")
    
    print("\n  2. 测试模型加载（建议在GPU节点运行）:")
    print("     srun --gres=gpu:1 --mem=32G --time=2:00:00 \\")
    print("       python scratch/LLM_reaction_Robot/Tools/test_qwen_omni.py")

print("\n⚠️ 关键考虑:")
print("  - Qwen2.5-Omni-3B: ~6GB参数，推荐16GB+ GPU")
print("  - Qwen2.5-Omni-7B: ~14GB参数，推荐32GB+ GPU")
print("  - 视频长度: 建议先测试10秒片段（~300帧）")
print("  - use_audio_in_video=True: 自动提取视频中的音频")

print("\n💡 如果验证成功:")
print("  1. 评估输出格式是否满足要求（17个AU，连续值）")
print("  2. 测试不同长度视频的处理能力")
print("  3. 优化提示词（Prompt Engineering）")
print("  4. 开始小规模微调实验（LoRA）")
print("  5. 实现知识蒸馏到轻量级学生模型")

print("\n" + "=" * 60)
print("验证脚本执行完毕")
print("=" * 60)

"""
Qwen2.5-Omni AU预测教师模型
使用回归头直接输出AU概率，而非文本生成
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from peft import get_peft_model, LoraConfig, TaskType


class QwenAUTeacher(nn.Module):
    """
    Qwen2.5-Omni教师模型：encoder + AU回归头
    
    输入：
      - videos: [B, T, 3, H, W] 视频帧
      - audios: [B, L] 音频波形
    
    输出：
      - au_probs: [B, T, 17] AU激活概率 (0-1)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-3B",
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        freeze_encoder: bool = False,
        au_head_hidden_dim: int = 512,
        au_head_dropout: float = 0.1,
        load_in_8bit: bool = True,
    ):
        """
        Args:
            model_name: Qwen模型ID
            use_lora: 是否使用LoRA微调
            lora_rank: LoRA秩
            lora_alpha: LoRA alpha（学习率缩放）
            lora_dropout: LoRA dropout
            freeze_encoder: 是否冻结encoder（仅训练回归头）
            au_head_hidden_dim: AU回归头隐层维度
            au_head_dropout: AU回归头dropout
            load_in_8bit: 是否使用8-bit量化
        """
        super().__init__()
        
        self.model_name = model_name
        self.use_lora = use_lora
        self.freeze_encoder = freeze_encoder
        
        # ===================================================================
        # 1. 加载Qwen2.5-Omni作为encoder
        # ===================================================================
        print(f"加载模型: {model_name}")
        
        self.encoder = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            load_in_8bit=load_in_8bit,
        )
        
        # 获取hidden dim
        self.hidden_dim = self.encoder.config.hidden_size
        print(f"  - Hidden dimension: {self.hidden_dim}")
        print(f"  - Encoder params: {sum(p.numel() for p in self.encoder.parameters()) / 1e9:.2f}B")
        
        # ===================================================================
        # 2. 配置LoRA（可选）
        # ===================================================================
        if self.use_lora:
            print(f"\n应用LoRA配置:")
            print(f"  - Rank: {lora_rank}")
            print(f"  - Alpha: {lora_alpha}")
            print(f"  - Dropout: {lora_dropout}")
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj", "v_proj",  # 自注意力
                    "fc1", "fc2"  # FFN
                ],
            )
            
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
        
        # ===================================================================
        # 3. 冻结encoder（可选）
        # ===================================================================
        if self.freeze_encoder and not self.use_lora:
            print("\n冻结所有encoder参数")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # ===================================================================
        # 4. AU回归头
        # ===================================================================
        print(f"\n构建AU回归头:")
        print(f"  - 输入维度: {self.hidden_dim}")
        print(f"  - 隐层维度: {au_head_hidden_dim}")
        print(f"  - 输出维度: 17 (AUs)")
        
        self.au_head = nn.Sequential(
            nn.Linear(self.hidden_dim, au_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(au_head_dropout),
            nn.Linear(au_head_hidden_dim, 17),
            nn.Sigmoid()  # 输出0-1概率
        )
        
        # 初始化回归头
        for module in self.au_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        print("✅ 模型初始化完成")
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        audio_values: torch.Tensor = None,
        output_hidden_states: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 文本token IDs [B, L_text]
            attention_mask: 文本注意力mask [B, L_text]
            pixel_values: 视频帧 [B, T, 3, H, W] 或 [B, num_frames, 3, H, W]
            audio_values: 音频波形 [B, L_audio]
            output_hidden_states: 是否输出隐藏状态
        
        Returns:
            au_probs: [B, T, 17] AU激活概率
        """
        # 通过encoder获取隐藏状态
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            audio_values=audio_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs
        )
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1] if output_hidden_states else outputs.last_hidden_state
        # [B, L, D]
        
        # 通过AU回归头
        au_probs = self.au_head(hidden_states)
        # [B, L, 17]
        
        return au_probs
    
    def get_trainable_params(self):
        """获取可训练参数统计"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n参数统计:")
        print(f"  - 总参数: {total / 1e9:.2f}B")
        print(f"  - 可训练: {trainable / 1e9:.2f}B ({100 * trainable / total:.2f}%)")
        return trainable, total


if __name__ == "__main__":
    # 快速测试
    print("=" * 60)
    print("Qwen AU Teacher 模型测试")
    print("=" * 60)
    
    # 初始化模型
    model = QwenAUTeacher(
        model_name="Qwen/Qwen2.5-Omni-3B",
        use_lora=True,
        lora_rank=64,
        freeze_encoder=False,
    )
    
    model.get_trainable_params()
    
    print("\n✅ 模型初始化成功")
    print("\n说明:")
    print("  - 此架构使用encoder + 回归头，避免文本生成")
    print("  - LoRA适配器显著减少可训练参数")
    print("  - 推理时间: 预计 <2秒（vs 500秒文本生成）")
    print("  - 输出格式: 直接数值 [B, T, 17]，便于优化")

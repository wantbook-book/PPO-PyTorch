
import torch
import torch.nn as nn

class StrengthsPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super(StrengthsPredictor, self).__init__()
        
        # 如果没有指定hidden_dim，使用自适应策略
        if hidden_dim is None:
            # 对于strength prediction任务，使用倒置金字塔
            # 因为我们要从高维hidden state压缩到低维strength值
            if input_dim > 512:
                hidden_dim = input_dim // 4  # 大模型用更大的压缩比
            else:
                hidden_dim = input_dim // 2  # 小模型用较小的压缩比
            
            # 确保hidden_dim不会太小
            hidden_dim = max(hidden_dim, output_dim * 4, 32)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # Output strengths score between 0 and 1
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization for better training stability"""
        layers = list(self.network.children())
        for i, module in enumerate(layers):
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for hidden layers
                nn.init.xavier_uniform_(module.weight)
                
                # 特殊处理最后一层（输出层）
                if i == len(layers) - 2:  # 最后一个Linear层（Sigmoid前）
                    # 使用更小的权重初始化，让Sigmoid输出更接近0.5
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        # 初始化bias为0，让初始输出接近0.5
                        nn.init.constant_(module.bias, 0.0)
                else:
                    # 隐藏层bias初始化为小正值避免死神经元
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state):
        # 如果类型不同，转化为参数类型
        if state.dtype != torch.float32:
            state = state.float()
        return self.network(state)


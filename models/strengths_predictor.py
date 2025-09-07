
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
            # nn.Sigmoid()  # Output strengths score between 0 and 1
            nn.Softmax(dim=-1)  # Output probability distribution over strengths
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights to produce outputs close to 0 with diversity"""
        layers = list(self.network.children())
        for i, module in enumerate(layers):
            if isinstance(module, nn.Linear):
                # 特殊处理最后一层（输出层）
                if i == len(layers) - 2:  # 最后一个Linear层（Sigmoid前）
                    # 使用小的权重初始化增加多样性
                    nn.init.normal_(module.weight, mean=0.0, std=0.1)
                    if module.bias is not None:
                        # 初始化bias为负值，让Sigmoid输出接近0
                        # 添加小的随机噪声增加多样性
                        bias_init = -2.0 + torch.randn(module.bias.shape) * 0.2
                        nn.init.constant_(module.bias, 0.0)
                        module.bias.data = bias_init
                else:
                    # 隐藏层使用Xavier初始化但添加噪声
                    nn.init.xavier_uniform_(module.weight)
                    # 添加小的随机噪声到权重
                    module.weight.data += torch.randn_like(module.weight) * 0.01
                    
                    # 隐藏层bias初始化为小的随机值
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.02, 0.02)
    
    def forward(self, state):
        # 如果类型不同，转化为参数类型
        if state.dtype != torch.float32:
            state = state.float()
        return self.network(state)


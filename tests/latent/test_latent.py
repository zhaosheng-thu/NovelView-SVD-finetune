import torch
import torch.nn as nn

class LinearProjectionAdapter(nn.Module):
    def __init__(self, input_channels, depth, height, width):
        super(LinearProjectionAdapter, self).__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.fc = nn.Linear(height * width, depth * height * width)

    def forward(self, x):
        # x: (batch_size, C, H, W)
        batch_size, C, H, W = x.size()
        assert H == self.height and W == self.width, "Input height and width must match the initialized dimensions."

        # 将 (batch_size, C, H, W) 变为 (batch_size, C, H*W)
        x = x.view(batch_size, C, -1)

        # 线性投影：将 (batch_size, C, H*W) 变为 (batch_size, C, depth*H*W)
        x = self.fc(x)

        # 将 (batch_size, C, depth*H*W) 变为 (batch_size, C, depth, H, W)
        x = x.view(batch_size, C, self.depth, self.height, self.width)

        return x

# 示例用法
input_tensor = torch.randn(8, 64, 32, 32)  # 示例输入 (batch_size, C, H, W)
model = LinearProjectionAdapter(input_channels=64, depth=5, height=32, width=32)
output_tensor = model(input_tensor)
print(output_tensor.shape)  # 预期形状: torch.Size([8, 64, 4, 32, 32])

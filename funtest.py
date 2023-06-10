"""
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/5/14 17:14   xxx      1.0         None
"""


import torch

x = torch.Tensor([[  1,  2,  3,  4,  5],
                  [  6,  7,  8,  9, 10],
                  [ 11, 12, 13, 14, 15],
                  [ 16, 17, 18, 19, 20],
                  [ 21, 22, 23, 24, 25]])
x = x.unfold(0, 3, 2)
print(x)
print(x.size())
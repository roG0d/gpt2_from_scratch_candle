
import torch
import torch.nn as nn

logits_2 = torch.tensor([[1,1,1],[2,2,2],[3,3,3]])

print(logits_2)
print(logits_2.shape)

logits_3 = torch.tensor([[[0, 0], [1, 1]],
                          [[0, 0], [1, 1]],
                          [[0, 0], [1, 1]]])

print(logits_3)
print(logits_3.shape)

# Remove the second dimension 3x2x2 -> 3x2
slice = logits_3[:, -1, :] # becomes (B, C)
print(slice)
print(slice.shape)

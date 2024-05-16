
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


# Multinomial -> take a sample from a given multinomial distribution, in this case, the one obtained with
# softmax (that, given a tensor, return a distribution over a dimension)
# row[0] -> most likely class 0,  row[1] -> most likely class 1,  row[2] -> most likely class 2
logits_2 = torch.tensor([[100.0,1.0,1.0], [1.0,100.0,1.0],[1.0,1.0,100.0]],dtype=float)
softmax = torch.softmax(logits_2,1)
next_idx = torch.multinomial(softmax,1)
print(softmax)
print(next_idx)


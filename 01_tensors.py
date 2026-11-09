import torch
a = torch.tensor([[1, 2], [3, 4]])
b = torch.ones(2, 3)
c = torch.rand(2, 3)
print(a, b, c)

print(a.shape, a.dtype, a.device)
print(b.shape, b.dtype, b.device)

b_gpu = b.to("cuda")
print(b_gpu.device)


x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x)

print(x[0])
print(x[:, 0])
print(x[1, 2])
print(x[:, 1])
print(x[:, 1:])

m = torch.tensor([[1, 2], [3, 4]])
print(m * m)
print(m @ m)
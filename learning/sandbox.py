from __future__ import print_function
import torch

x = torch.Tensor(5, 3)

print(x)

x = torch.rand(5,3)

print(x)

print(x.size())

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)

print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# let us run this cell only if CUDA is available
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y

x

#############

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

gradients = torch.FloatTensor([0.001])
out.backward(gradients)

print(x.grad)

x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.001])
gradients = torch.FloatTensor([1,1,1])
y.backward(gradients)

print(x.grad)
print(x.grad)

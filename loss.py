import torch
import torch.nn as nn

class SoftmaxT(nn.Module):
  def __init__(self, T=1):
    super(SoftmaxT, self).__init__()
    self.T = T

  # x: [batchsize, class_num]
  def forward(self, x):
    x = x / self.T
    x = torch.exp(x)
    x = torch.div(x, torch.unsqueeze(torch.sum(x, dim=1), 1))
    return x

class Softloss(nn.Module):
  def __init__(self, T):
    super(Softloss, self).__init__()
    self.softmax_big = SoftmaxT(T)
    self.softmax_small = SoftmaxT(T)
  
  # v: big model's logits 
  # z: small model's logits
  def forward(self, v, z):
    p = self.softmax_big(v)
    q = self.softmax_small(z)
    return -torch.sum(p * torch.log(q)) / v.shape[0]

class KDloss(nn.Module):
  def __init__(self, T, alpha):
    super(KDloss, self).__init__()
    self.softloss = Softloss(T)
    self.hardloss = nn.CrossEntropyLoss()
    self.T = T
    self.alpha = alpha

  def forward(self, input_small, input_big, target):
    softloss = self.softloss(input_big, input_small)*self.alpha*self.T*self.T
    hardloss = self.hardloss(input_small, target)*(1-self.alpha)
    return softloss, hardloss, softloss+hardloss

if __name__ == "__main__":
  softmax_t = SoftmaxT(1)
  x = torch.randn(64, 100)
  output_t = softmax_t(x)
  print(output_t)
  softmax = nn.Softmax(dim=1)
  output = softmax(x)
  print(output)
  print(torch.allclose(output_t, output))

  softmax_t_1 = SoftmaxT(1)
  softmax_t_2 = SoftmaxT(2)
  softmax_t_10 = SoftmaxT(10)
  x = torch.tensor([[0.1, 0.1, 0.9, 0.1]])
  output = softmax_t(x)
  print(output)
  output = softmax_t_2(x)
  print(output)
  output = softmax_t_10(x)
  print(output)

  v = torch.tensor([[0.1, 0.1, 0.9, 0.1], [0, 0.5, 0.5, 0.9]])
  z = torch.tensor([[0.1, 0.1, 0.9, 0.1], [0, 0.5, 0.5, 0.9]])
  softloss = Softloss(1)
  output = softloss(v, z)
  print(output)
  criterion = nn.CrossEntropyLoss()
  target = softmax_t_1(v)
  output = criterion(z, target)
  print(output)

  small = torch.tensor([[-1., -1, 4, -1], [-1, -1, -1, 7.]])
  # big = torch.tensor([[0.1, 0.1, 0.9, 0.1], [0.2, 0.1, 0.3, 0.9]])
  big = torch.tensor([[-10, -10, 10, -10], [-10, -10, -10, 10.]])
  target = torch.tensor([2, 3])
  KD_loss = KDloss(3, 0.9)
  softloss, hardloss, totalloss = KD_loss(small, big, target)
  print(softloss, hardloss, totalloss)

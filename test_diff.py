import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = 10
        self.x = torch.ones(size=[self.T])                
        self._eps = distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.mu = nn.Parameter(torch.tensor(3.0))
        self.mu.requires_grad = True
        
    def forward(self):
        x_t = torch.clone(self.x)
        for t in range(1, self.T):
            epsilons = self._eps.rsample((self.T,))
            x_t[t] = self.mu*self.x[t-1]
            
        return x_t
    
model = TestModel()

gt = torch.tensor([13.0])

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss = torch.nn.MSELoss()
print(model.state_dict().items())

for i in range(3):
    print(f"Iteration {i}")
    optimizer.zero_grad()
    y = model()
    l = loss(y.sum(), gt)
    l.backward(retain_graph=False)
    optimizer.step()
    print(model.state_dict().items())

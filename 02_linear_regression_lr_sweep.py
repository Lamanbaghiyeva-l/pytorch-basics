import torch
import torch.nn as nn

xs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(1)
ys = torch.tensor([3.0, 5.0, 7.0, 9.0, 11.0]).unsqueeze(1)
loss_fn = nn.MSELoss()

for lr in [10,3,1,0.1, 0.01, 0.001, 0.0001]:
    torch.manual_seed(0)                 # eyni başlanğıc

    model = nn.Linear(1, 1)              # hər lr üçün YENİ model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # YENİ Adam optimizer, bu lr ilə

    for step in range(300):
        optimizer.zero_grad()            # 1. sıfırla
        y_pred = model(xs)               # 2. forward
        loss = loss_fn(y_pred, ys)       # 3. loss
        loss.backward()                  # 4. backward
        optimizer.step()                 # 5. step

    print(f"lr={lr}  final loss={loss.item():.6f}  w={model.weight.item():.3f}  b={model.bias.item():.3f}")
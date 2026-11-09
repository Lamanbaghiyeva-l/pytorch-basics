import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# ---------- 1. DATA ----------
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

training_data = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_data     = datasets.MNIST(root="data", train=False, download=True, transform=transform)

training_loader = DataLoader(training_data, batch_size=64, shuffle=True)
test_loader     = DataLoader(test_data,     batch_size=64, shuffle=False)

# ---------- 2. CİHAZ ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# ---------- 3. MODEL: CNN ----------
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),    # [64, 16, 28, 28]
            nn.ReLU(),                                     # ✅ ReLU (böyük U)
            nn.MaxPool2d(2),                               # ✅ MaxPool2d → [64, 16, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # ✅ 16 kanal girir → [64, 32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → [64, 32, 7, 7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # → [64, 32*7*7] = [64, 1568]
            nn.Linear(32 * 7 * 7, 128),                    # ✅ 1568 → 128
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)          # naxışları tap
        x = self.classifier(x)    # qərar ver
        return x

torch.manual_seed(0)
model = MNISTCNN().to(device)
print("parametr sayı:", sum(p.numel() for p in model.parameters()))




# ---------- 4. ÖYRƏTMƏK ----------
for lr in (0.001, 0.01, 0.1, 1):
    torch.manual_seed(0)
    model = MNISTCNN().to(device)                              # hər lr üçün YENİ model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   # ✅ torch.optim, parametrlər, lr ədəddir

    for epoch in range(10):
        model.train()                                             # öyrənmə rejimi
        for images, labels in training_loader:
            images, labels = images.to(device), labels.to(device) # ✅ hər ikisi GPU-ya

            optimizer.zero_grad()                                 # ✅ 1. sıfırla
            outputs = model(images)                               #    2. forward
            loss = loss_fn(outputs, labels)                       # ✅ 3. loss: proqnoz və düzgün cavab
            loss.backward()                                       # ✅ 4. backward: mötərizə ilə!
            optimizer.step()                                      # ✅ 5. step

        print(f"lr {lr}, epoch {epoch + 1}, son loss: {loss.item():.4f}")

# ---------- 5. TEST: ACCURACY ----------
    model.eval()                                                  # yoxlama rejimi
    correct = 0
    total = 0
    with torch.no_grad():                                         # yoxlamada gradient lazım deyil
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)                   # hər şəkil üçün ən yüksək bal
            correct += (predictions == labels).sum().item()       # neçəsi düzdür
            total += labels.size(0)                               # neçə şəkil yoxlandı

    print(f"Test accuracy: {100 * correct / total:.2f}%")
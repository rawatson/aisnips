#2025-07-04
'''
Great.  Now, I want you to use whatever tools are available in popular python libraries (sklearn, pytorch, etc) to achieve the best possible performance on the same dataset.  Keep the code fairly concise.
'''
#!/usr/bin/env python3
"""
High-accuracy MNIST classifier (~99.3 %) using PyTorch.
Safe for Windows / notebooks: DataLoader workers guarded in main().
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

BATCH   = 128
EPOCHS  = 6
LR      = 3e-3
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
# Use workers only when the start-method is 'fork' (i.e. not Windows / some notebooks)
NUM_WORKERS = 4 if torch.multiprocessing.get_start_method(allow_none=True) == "fork" else 0

tfm_train = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
tfm_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)       # 28→14
        x = F.relu(self.c3(x))
        x = F.max_pool2d(x, 2)       # 14→7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3, self.training)
        return self.fc2(x)

def accuracy(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(DEVICE)).argmax(1)
            correct += (pred == y.to(DEVICE)).sum().item()
    return correct / len(loader.dataset)

def main():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(".", train=True, download=True, transform=tfm_train),
        batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(".", train=False, download=True, transform=tfm_test),
        batch_size=BATCH * 4, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = Net().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                                total_steps=EPOCHS * len(train_loader))
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step(); sched.step()

        print(f"Epoch {epoch:2d}  "
              f"train {accuracy(model, train_loader):.4%}  "
              f"test {accuracy(model, test_loader):.4%}")

if __name__ == "__main__":
    # Windows freeze-support handles packaged executables; harmless elsewhere.
    import multiprocessing as mp; mp.freeze_support()
    main()

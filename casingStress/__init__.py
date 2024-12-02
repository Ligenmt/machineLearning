import torch

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 3)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        # softmax做归一化
        # x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        # x = torch.nn.functional.softmax(self.fc3(x), dim=1)
        x = self.fc4(x)
        return x
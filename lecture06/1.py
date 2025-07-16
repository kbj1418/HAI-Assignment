import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(17, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.seq(x)

myFirstModel = MyModel()

loss_fn = nn.MSELoss()
optimizer = optim.SGD(myFirstModel.parameters(), lr=0.01)

epchs = 100

for epch in range(1, epchs + 1):
    inputs = torch.randn(1, 17)
    labels = torch.rand(1, 1)

    optimizer.zero_grad()
    outputs = myFirstModel(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"epch: {epch}, loss: {loss.item():.4f}")


img = Image.open('HAI-Assignment/lecture06/numberImage.png').convert('L')
to_tensor = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
x = to_tensor(img).unsqueeze(0)  

myFirstModel.eval()
with torch.no_grad():
    logits = myFirstModel(torch.randn(1, 17))  
print(logits)

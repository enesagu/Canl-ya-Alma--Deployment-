# create_model.py

import torch

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        # Modelinizi burada tanımlayın
        return x * 2

# Örnek giriş verisi oluşturun
input_data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).float()

# Modeli oluşturun, eğitin ve kaydedin
model = SimpleModel()
output = model(input_data)
torch.save(model.state_dict(), "simple_model.pth")

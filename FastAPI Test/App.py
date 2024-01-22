from fastapi import FastAPI
from pydantic import BaseModel
import torch
import resource
import os

# FastAPI uygulama oluşturulması
app = FastAPI()

# Kaydedilmiş modelin yüklenmesi
class MyModel:
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            # Modelinizi burada tanımlayın
            return x * 2

    def __init__(self):
        self.model = self.SimpleModel()
        self.model.load_state_dict(torch.load("simple_model.pth"))
        self.model.eval()

# FastAPI path parametre model input'u alması
class Item(BaseModel):
    input_data: list

# Sistem izleme fonksiyonu
def get_resource_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print("CPU Time:", usage.ru_utime + usage.ru_stime, "seconds")
    print("Memory Usage:", (usage.ru_maxrss / 1024), "KB")

# FastAPI endpoint tanımı
@app.post("/predict")
def predict(item: Item):
    # Uygulama çalıştırıldığında sistem izleme fonksiyonunu çağırabilirsiniz.
    get_resource_usage()

    input_data = torch.tensor(item.input_data).float()
    
    # Kaydedilmiş modelin kullanılması
    my_model = MyModel()
    with torch.no_grad():
        output = my_model.model(input_data)

    return {"prediction": output.numpy().tolist()}

# Uygulama çalışırken kaynak kullanımını izlemek için
if __name__ == "__main__":
    # Uygulamayı çalıştırma
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

from flask import Flask, request, jsonify
import torch
import resource
import os

app = Flask(__name__)

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        # Modelinizi burada tanımlayın
        return x * 2

class MyModel:
    def __init__(self):
        self.model = SimpleModel()
        self.model.load_state_dict(torch.load("simple_model.pth"))
        self.model.eval()

def get_resource_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print("CPU Time:", usage.ru_utime + usage.ru_stime, "seconds")
    print("Memory Usage:", (usage.ru_maxrss / 1024), "KB")

@app.route("/predict", methods=["POST"])
def predict():
    get_resource_usage()

    input_data = torch.tensor(request.json["input_data"]).float()

    my_model = MyModel()
    with torch.no_grad():
        output = my_model.model(input_data)

    return jsonify({"prediction": output.numpy().tolist()})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)

import streamlit as st
import torch
import resource

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
    st.write("CPU Time:", usage.ru_utime + usage.ru_stime, "seconds")
    st.write("Memory Usage:", (usage.ru_maxrss / 1024), "KB")

def main():
    st.title("Streamlit Model Inference")

    # Sayfanın sol tarafında bir sidebar oluşturun
    st.sidebar.header("Options")

    # Input verisini girmek için kullanıcıya bir alan sağlayın
    input_data = st.sidebar.text_input("Enter input data (comma-separated):", "1,2,3,4,5")

    # Kullanıcının girdiği veriyi sayılara dönüştürün
    input_data = list(map(float, input_data.split(',')))

    # "Predict" butonuna basıldığında modeli çalıştırın
    if st.sidebar.button("Predict"):
        get_resource_usage()

        # Girdiyi tensora çevirin
        input_tensor = torch.tensor(input_data).float()

        # Modeli oluşturun ve girdi verisini tahmin edin
        my_model = MyModel()
        with torch.no_grad():
            output = my_model.model(input_tensor)

        # Tahmin sonucunu gösterin
        st.success(f"Prediction: {output.numpy().tolist()}")

if __name__ == "__main__":
    main()

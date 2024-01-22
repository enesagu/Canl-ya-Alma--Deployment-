import resource
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import SimpleModel
import torch

def get_resource_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print("CPU Time:", usage.ru_utime + usage.ru_stime, "seconds")
    print("Memory Usage:", (usage.ru_maxrss / 1024), "KB")

class MyModel:
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            # Modelinizi burada tanımlayın
            return x * 2

    def __init__(self):
        self.model = self.SimpleModel()
        self.model.load_state_dict(torch.load("simple_model.pth"))
        self.model.eval()

@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):
    def post(self, request, *args, **kwargs):
        # Sistem izleme fonksiyonunu çağır
        get_resource_usage()

        input_data = torch.tensor(request.POST.getlist('input_data')).float()

        my_model = MyModel()
        with torch.no_grad():
            output = my_model.model(input_data)

        return JsonResponse({"prediction": output.numpy().tolist()})

class PredictAPIView(APIView):
    def post(self, request, *args, **kwargs):
        # Sistem izleme fonksiyonunu çağır
        get_resource_usage()

        input_data = torch.tensor(request.data.get('input_data')).float()

        my_model = MyModel()
        with torch.no_grad():
            output = my_model.model(input_data)

        return Response({"prediction": output.numpy().tolist()})

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO: Calcular dimension de salida
        out_dim = self.calc_out_dim(input_dim,3)

        # TODO: Define las capas de tu red
        self.conv1 = nn.Conv2d(1,8,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8,4,3)
        self.fc1 =nn.Linear(400, 64)
        self.fc2 =nn.Linear(64, n_classes)
        self.softmax =nn.Softmax(dim=1)
        self.to(self.device)
 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        #Flatten
        x = x.view(-1,400)
        x = self.fc1(x)
        logits = self.fc2(x)
        proba = self.softmax(logits)       
        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        '''
            Guarda el modelo en el path especificado
            args:
            - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
            - path (str): path relativo donde se guardará el modelo
        '''
        models_path = file_path / 'models' / model_name
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        '''
            Carga el modelo en el path especificado
            args:
            - path (str): path relativo donde se guardó el modelo
        '''
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / 'models' / model_name
        self.load_state_dict(torch.load(models_path, map_location=torch.device('cpu')))

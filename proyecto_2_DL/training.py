from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_loader
from network import Network
from plot_losses import PlotLosses

def validation_step(val_loader, net, cost_function):
    '''
        Realiza un epoch completo en el conjunto de validación
        args:
        - val_loader (torch.DataLoader): dataloader para los datos de validación
        - net: instancia de red neuronal de clase Network
        - cost_function (torch.nn): Función de costo a utilizar

        returns:
        - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    '''
    val_loss = 0.0
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch['transformed']
        batch_labels = batch['label']
        device = net.device
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            # logits, probs = net(batch_imgs)
            probs = torch.empty(batch_labels.shape[0],7)
            logits = torch.empty(batch_labels.shape[0],7)
            for j in range(0,batch_labels.shape[0]):    
                logits[j], probs[j] = net(batch_imgs[j])
            # print(probs)
            predictions = torch.argmax(probs,dim=1)
            # print(predictions)
            num_classes = 7  # Number of classes
            target = nn.functional.one_hot(batch_labels, num_classes=num_classes).float()
            loss = cost_function(probs,target)
            val_loss += loss
    # TODO: Regresa el costo promedio por minibatch
    return val_loss/i

def train():
    # Hyperparametros
    learning_rate = 1e-4
    n_epochs=100
    batch_size = 256

    # Train, validation, test loaders
    train_dataset, train_loader = \
        get_loader("train",
                    batch_size=batch_size,
                    shuffle=True)
    val_dataset, val_loader = \
        get_loader("val",
                    batch_size=batch_size,
                    shuffle=False)
    print(f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}")

    plotter = PlotLosses()
    # Instanciamos tu red
    modelo = Network(input_dim = 48,
                     n_classes = 7)

    # TODO: Define la funcion de costo
    criterion = nn.CrossEntropyLoss()

    # Define el optimizador
    optimizer = optim.Adam(modelo.parameters(),
                       lr=learning_rate)

    best_epoch_loss = np.inf
    for epoch in range(n_epochs):
        train_loss = 0
        running_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch['transformed']
            # print(batch_imgs.shape)
            batch_labels = batch['label']
            # TODO Zero grad, forward pass, backward pass, optimizer step
            optimizer.zero_grad()
            probs = torch.empty(batch_labels.shape[0],7)
            logits = torch.empty(batch_labels.shape[0],7)
            for j in range(batch_labels.shape[0]):    
                logits[j], probs[j] = modelo(batch_imgs[j])  
            # print(probs)
            # predictions = torch.argmax(probs,dim=1)
            # predictions = probs
            # print(predictions)
            # print(type(predictions))
            # print(type(batch_labels))
            # print(batch_labels)
            # target = torch.zeros((batch_labels.shape[0],7), dtype=torch.float)
            # for j in range(batch_labels.shape[0]):    
            #     target[j][batch_labels[j]] = 1.0
            num_classes = 7  # Number of classes
            target = nn.functional.one_hot(batch_labels, num_classes=num_classes).float()
            loss = criterion(probs, target)
            loss.backward()
            optimizer.step()

            # TODO acumula el costo
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                train_loss += running_loss
                running_loss = 0.0
        print(i)
        # TODO Calcula el costo promedio
        train_loss = train_loss/i
        val_loss = validation_step(val_loader, modelo, criterion)
        tqdm.write(f"Epoch: {epoch}, train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

        # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
        if val_loss<best_epoch_loss:
            modelo.save_model("prueba1.pth")
        plotter.on_epoch_end(epoch, train_loss, val_loss)
    plotter.on_train_end()

if __name__=="__main__":
    train()
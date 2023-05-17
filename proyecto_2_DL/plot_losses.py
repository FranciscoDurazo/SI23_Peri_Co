import datetime as dt
import pathlib
import matplotlib
import matplotlib.pyplot as plt
file_path = pathlib.Path(__file__).parent.absolute()

class PlotLosses():
    def __init__(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, train_loss, val_loss):        
        self.x.append(self.i)
        self.losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.i += 1
        plt.cla()
        plt.plot(self.x, self.losses, label="Costo de entrenamiento promedio")
        plt.plot(self.x, self.val_losses, label="Costo de validación promedio")
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show(block=False)
        plt.pause(5)

    def on_train_end(self):
        plt.show()
        today = dt.datetime.now().strftime("%Y-%m-%d")
        losses_file = file_path/ f'figures/losses_{today}.png'
        plt.savefig(losses_file)
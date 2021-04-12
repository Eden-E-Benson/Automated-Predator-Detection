import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Plotting():
    def plot_losses(self, training_losses, validation_losses):
        plt.figure()
        epochs = range(len(training_losses))
        line1 = plt.plot(epochs, training_losses, label = "Training Loss")
        line2 = plt.plot(epochs, validation_losses, label = "Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.legend(loc = "upper right")
        plt.title("Training/Validation Loss Comparison")
        plt.savefig("plots/training_validation_loss_comp.png")
        plt.show()

    def stats_table(self, metrics):
        table = pd.DataFrame(columns = list(metrics[0].keys()))
        for metric in range(len(metrics)):
            table = table.append(metrics[metric], ignore_index = True)
        return table
    
    def visualise_images(self, images, labels, class_labels, fig_size = (25, 15), 
                         row_num = 5, col_num = 35):
        fig = plt.figure(figsize = fig_size)

        for i in np.arange(col_num):
            ax = fig.add_subplot(row_num, int(np.ceil(col_num / row_num)), i + 1, 
                                 xticks = [], yticks = [])
            self._imshow(images[i])
            ax.set_title(class_labels[labels[i]])
        
    def _imshow(self, image):
        image = image / 2 + 0.5 #unnormalise image
        plt.imshow(np.transpose(image, (1, 2, 0)))
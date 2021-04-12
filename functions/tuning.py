import numpy as np
import pandas as pd

from functions.model import Classifier
from functions.utilities import Utilities

import torch
import torchvision.models as models

class Tuning():
    def __init__(self):
        self.utils = Utilities()
        self.device = self.utils.select_processing_unit()
    
    def create_model(self, class_num, hidden_layer_sizes):
        # Create inception model
        inception = models.inception_v3(pretrained=True, aux_logits=True)
        
        classifier = Classifier(inception.fc.in_features, class_num, hidden_layer_sizes)
        
        # Prevents re-training of network by freezing (overwrites transfer learning)
        for parameter in inception.parameters():
            parameter.requires_grad = False
            
        inception.fc = classifier
        return inception
    
    def save_best_model(self, model_stats, class_num):
        idx = model_stats['Accuracy'].idxmax()
        info = list(model_stats.iloc[idx, :])
        model_name = info[0]
        hidden_layers = list(map(int, info[0].split("_")[2:]))
        
        model = self.create_model(class_num, hidden_layers)
        load_file_path = f"models/{model_name}.pt"
        save_file_path = f"models/best_model"
        
        for size in hidden_layers:
            save_file_path += f"_{str(size)}"
        save_file_path += ".pt"
        
        stats = model_stats.iloc[idx, 1:].to_dict()
        
        # Load model training and validation losses
        checkpoint = torch.load(load_file_path)
        model.load_state_dict(checkpoint["model_weights"])
        training_losses = checkpoint["training_losses"]
        validation_losses = checkpoint["validation_losses"]
        
        # Save model information again
        save_dict = {"model_weights": model.state_dict(),
                     "training_losses": training_losses,
                     "validation_losses": validation_losses,
                     "metrics": stats,
                     "model_name": model_name}
        torch.save(save_dict, save_file_path)
        
        print("Model saved -", end=" ")
        print(f"Name: '{model_name}' | Hidden layer sizes: '{hidden_layers}'")
        return model_name, hidden_layers
        
    def load_best_model(self, model, file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint["model_weights"])
        metrics = checkpoint["metrics"]
        model_name = checkpoint["model_name"]
        training_losses = checkpoint["training_losses"]
        validation_losses = checkpoint["validation_losses"]
        
        print(f"Model loaded. Name: '{model_name}'")
        return model_name, metrics, training_losses, validation_losses
    
    def get_model_data(self, model, test_loader):
        model.to(self.device)
        all_probs = torch.tensor([])
        targets = torch.tensor([])
        predictions = torch.tensor([])

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output, _ = model.forward(images)
                probabilities = torch.exp(output)
                top_preds = probabilities.max(dim = 1)[1]

                all_probs = torch.cat((all_probs.to(self.device), probabilities), dim = 0)
                targets = torch.cat((targets.to(self.device), labels), dim = 0)
                predictions = torch.cat((predictions.to(self.device), top_preds), dim = 0)

        return all_probs.cpu(), targets.cpu(), predictions.cpu()
    
    def indices_to_labels(self, predictions, labels, class_labels):
        class_labels_indices = range(len(class_labels))
        predictions = np.array(predictions, dtype=object)
        labels = np.array(labels, dtype=object)
        
        for i in range(len(class_labels_indices)):
            for j in range(len(predictions)):
                if predictions[j] == class_labels_indices[i]:
                    predictions[j] = class_labels[i]
                if labels[j] == class_labels_indices[i]:
                    labels[j] = class_labels[i]
        
        return predictions, labels
        
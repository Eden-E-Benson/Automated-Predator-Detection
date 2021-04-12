from functions.model import Classifier
from functions.utilities import Utilities

import torchvision.models as models

class Tuning():
    def __init__(self):
        self.utils = Utilities()
    
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
        
        training_losses, validation_losses = self.utils.load_model(model, load_file_path)
        stats = model_stats.iloc[idx, 1:].to_dict()
        
        torch.save({
            "model_weights": model.state_dict(),
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "model_name": model_name,
            "metrics": stats
        }, save_file_path)
        
        print("Model saved.")
        print(f"Name: {model_name} | Hidden layer sizes: {hidden_layers}")
        return model_name, hidden_layers
        
    def load_best_model(self, model, file_path):
        training_losses, validation_losses = self.utils.load_model(model, file_path)
        checkpoint = torch.load(file_path)
        model_name = checkpoint['model_name']
        metrics = checkpoint['metrics']
        
        print("Model loaded.")
        print(f"Name: {model_name}")
        print(f"Metrics: {metrics}")
        
        return model_name, metrics, training_losses, validation_losses
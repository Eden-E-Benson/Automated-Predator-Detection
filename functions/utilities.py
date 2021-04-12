import numpy as np
import time

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

class Utilities():
    def augment_data(self, file_path):
        transform_list = transforms.Compose([
                                transforms.Resize(299),
                                transforms.CenterCrop(299),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomRotation(30),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                     std=(0.5, 0.5, 0.5))])

        dataset = torchvision.datasets.ImageFolder(file_path, transform = transform_list)
        return dataset
    
    def split_data(self, dataset, batch_size, split_size = 0.15, seed = 1):
        torch.manual_seed(seed)
        np.random.seed(seed)

        data_len = len(dataset)
        data_indicies = list(range(data_len))
        np.random.shuffle(data_indicies)
        split = int(np.floor(split_size * data_len))

        valid_idx = data_indicies[:split] # 893
        test_idx = data_indicies[split:split * 2] # 893
        train_idx = data_indicies[split * 2:] # 4173

        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        train_data_size = len(train_sampler)

        valid_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                                   sampler = valid_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                                  sampler = test_sampler)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                                   sampler = train_sampler)

        return valid_loader, test_loader, train_loader, train_data_size
    
    def save_model(self, model, file_path, training_losses, validation_losses):
        torch.save({
            "model_weights": model.state_dict(),
            "training_losses": training_losses,
            "validation_losses": validation_losses
        }, file_path)
        
        print("Model saved.")
    
    def load_model(self, model, file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint["model_weights"])
        training_losses = checkpoint["training_losses"]
        validation_losses = checkpoint["validation_losses"]
        
        print("Model loaded.")
        
        return training_losses, validation_losses

    def time_taken(self, time_difference):
        minute_as_seconds, seconds = divmod(time_difference, 60)
        hours, minutes = divmod(minute_as_seconds, 60)
        print(f"Total time taken: {hours:.2f}:{minutes:.2f}:{seconds:.2f}")

    def early_stop_check(self, valid_loss, patience, early_stop, counter):
        if counter >= patience:
            early_stop = True
            print("Early stopping threshold met. Training has been stopped.")

        return early_stop
    
    def select_processing_unit(self):
        if torch.cuda.is_available():
            device = "cuda"
            print("Device set to GPU.")
        else:
            device = "cpu"
            print("Device set to CPU.")
        return device
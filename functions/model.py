import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    """
    
    """
    def __init__(self, input_node, output_node, hidden_layer_nodes, dropout_prob = 0.3):
        super().__init__()
        
        # 2048, 6, [256, 512, 1028]
        # self.fc1 = nn.Linear(2048, 256)
        # self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(512, 1028)
        # self.out = nn.Linear(1028, 6)
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_node, hidden_layer_nodes[0])])
        layer_size = zip(hidden_layer_nodes[:-1], hidden_layer_nodes[1:])
        self.hidden_layers.extend([nn.Linear(layer1, layer2) for layer1, layer2 in layer_size])
        self.output_layer = nn.Linear(hidden_layer_nodes[-1], output_node)
        self.dropout_layer = nn.Dropout(dropout_prob)
        
    def forward(self, dataset):
        # 
        for layer in self.hidden_layers:
            dataset = F.relu(layer(dataset))
            dataset = self.dropout_layer(dataset)
        dataset = F.log_softmax(self.output_layer(dataset), dim = 1)
        return dataset
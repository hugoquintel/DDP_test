from torch import nn

# the classification layer to put at the end of a Pre-trained language model
class ClassificationLayers(nn.Module):
    def __init__(self, config, labels_to_ids):
        super().__init__()
        self.cls_layers = nn.Sequential(nn.Linear(config.hidden_size, len(labels_to_ids)))
    def forward(self, input):
        logit = self.cls_layers(input)
        return logit
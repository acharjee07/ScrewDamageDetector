import timm
import torch.nn as nn

class Effnet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b3_ns', num_classes = 2, pretrained=True):
        super(Effnet, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.in_features, num_classes))
        

    def forward(self, images):
        out=self.backbone(images)
        return out
    


effnet=Effnet(model_name='tf_efficientnet_b0_ns', num_classes = 2, pretrained=True)




from torchvision.models import resnet50
import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self, n_class):
        super(model, self).__init__()
        backbone_model = resnet50(pretrained=True)
        layer0 = [backbone_model.conv1, backbone_model.bn1, backbone_model.relu, backbone_model.maxpool]
        layer1_4 = [backbone_model.layer1, backbone_model.layer2, backbone_model.layer3, backbone_model.layer4]
        avgpool = backbone_model.avgpool

        self.layers = nn.Sequential(*layer0, *layer1_4, avgpool)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.fc = nn.Linear(2048, n_class, bias=False)

    def forward(self, X):
        avg_pool_X = self.layers(X)
        avg_pool_X = avg_pool_X.view(64, -1) # Flatten
        feat = self.bottleneck(avg_pool_X)
        if self.train():
            return avg_pool_X, self.fc(feat)
        else:
            return avg_pool_X, feat

# net = model(n_class=10)
# a = net(torch.rand(64, 3, 144, 288))

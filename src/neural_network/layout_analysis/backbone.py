import torch
import torch.nn
import torchvision
import torchvision.models as tvm


class Bottleneck(torch.nn.Module):

    def __init__(self, height_image, length_image, nb_batch):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d()
    








class CustomResNet(torch.nn.Module):

    def __init__(self):
        super(CustomResNet, self).__init__()
        self.stem_conv = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.stem_maxpoool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    def stem_forward(self, image_input):
        result = self.stem_conv(image_input)
        result = torch.nn.BatchNorm2d(64)(result)
        result = torch.nn.ReLU()(result)
        result = self.stem_maxpoool(result)
        return result
    

    def layer1(self, input):
        pass

class Backbone(torch.nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass






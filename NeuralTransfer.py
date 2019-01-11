from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import sys

# Use cuda if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Run Higher resolution if GPU is available
image_size = 512 if torch.cuda.is_available() else 256

content_loader = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()])
style_loader = transforms.Compose([transforms.ToTensor()])


# Load style and content image, resize style image based on content image
def image_loader(content_image_name, style_image_name):
    content_image = Image.open(content_image_name)
    content_image = content_loader(content_image).unsqueeze(0)
    content_image = content_image.to(device, torch.float)
    style_image = Image.open(style_image_name)
    style_image = style_image.resize((content_image.size()[2], content_image.size()[3]), Image.ANTIALIAS)
    style_image = style_loader(style_image).unsqueeze(0)
    style_image.to(device, torch.float)
    return content_image, style_image


content_image, style_image = image_loader(sys.argv[1], sys.argv[2])

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_gram_matrix(input):
    batch_size, number_of_feature_maps, width, height = input.size()
    features = input.view(batch_size * number_of_feature_maps, width * height)
    gram_product = torch.mm(features, features.t())
    return gram_product.div(batch_size * number_of_feature_maps * width * height)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = get_gram_matrix(target_feature).detach()

    def forward(self, input):
        gram_matrix = get_gram_matrix(input)
        self.loss = F.mse_loss(gram_matrix, self.target)
        return input


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def get_model(model_name):
    if model_name == "vgg16":
        content_layers = ['conv_4', 'conv_7', 'conv_13']
        style_layers = ['conv_1', 'relu_2', 'conv_3', 'relu_4', 'conv_5', 'relu_6', 'pool_7', 'conv_8',
                        'conv_9', 'relu_10', 'conv_11', 'relu_12', 'pool_13']
        return models.vgg16(pretrained=True).features.to(device).eval(), content_layers, style_layers
    else:
        content_layers = ['conv_4', 'conv_8', 'conv_12', 'conv_16']
        style_layers = ['conv_1', 'relu_2', 'conv_3', 'relu_4', 'conv_5', 'conv_6', 'relu_7', 'pool_8',
                        'conv_9', 'relu_10', 'conv_11', 'relu_12', 'conv_13', 'conv_14', 'relu_15', 'pool_16']
        return models.vgg19(pretrained=True).features.to(device).eval(), content_layers, style_layers


cnn, content_layers, style_layers = get_model(sys.argv[3])


def get_losses(cnn, normalization_mean, normalization_std, style_image, content_image,
               content_layers=content_layers, style_layers=style_layers):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    for cnn_layer in cnn.children():
        if type(cnn_layer) == nn.Conv2d:
            i += 1
            layer_name = 'conv_{}'.format(i)
        elif type(cnn_layer) == nn.ReLU:
            layer_name = 'relu_{}'.format(i)
            cnn_layer = nn.ReLU(inplace=False)
        elif type(cnn_layer) == nn.MaxPool2d:
            layer_name = 'pool_{}'.format(i)
        else:
            raise Exception('Unrecognized layer')

        model.add_module(layer_name, cnn_layer)

        if layer_name in content_layers:
            content_loss = ContentLoss(model(content_image).clone())
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if layer_name in style_layers:
            style_loss = StyleLoss(model(style_image).clone())
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if type(model[i]) != ContentLoss and type(model[i]) != StyleLoss:
            model = model[:i]
        else:
            break
    return model, style_losses, content_losses


input_img = content_image.clone()


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_image, style_image, input_image, num_steps=400,
                       style_weight=10000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_losses(cnn, normalization_mean, normalization_std,
                                                     style_image, content_image)
    optimizer = optim.LBFGS([input_image.requires_grad_()])

    print('Optimizing Image..')
    global epoch
    epoch = 0
    while epoch <= num_steps:

        def closure():
            input_image.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_image)
            style_score = 0.0
            content_score = 0.0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()

            global epoch
            epoch += 1
            if epoch % 25 == 0:
                print("run {}:".format(epoch))
                print('Style Loss : {:4f} Content Loss: {:4f}\n'.format(
                    style_score.item(), content_score.item()))
            return loss

        optimizer.step(closure)

    input_image.data.clamp_(0, 1)
    return input_image


output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_image, style_image, input_img)

image = output.cpu().clone()
image = image.squeeze(0)
unloader = transforms.ToPILImage()
image = unloader(image)
image.save('output.jpg')

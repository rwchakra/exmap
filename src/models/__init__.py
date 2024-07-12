import torch
import torchvision
import types


def _replace_fc(model, output_dim):
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, output_dim)
    return model


def _vgg_replace_fc(model, output_dim):
    model.fc = torch.nn.Identity()
    model.fc.in_features = model.classifier[0].in_features
    delattr(model, "classifier")

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    forwardType = types.MethodType
    model.forward = forwardType(forward, model)
    return _replace_fc(model, output_dim)


def _densenet_replace_fc(model, output_dim):
    model.fc = torch.nn.Identity()
    model.fc.in_features = model.classifier.in_features
    delattr(model, "classifier")

    def classifier(self, x):
        return self.fc(x)

    model.classifier = types.MethodType(classifier, model)
    return _replace_fc(model, output_dim)


def imagenet_resnet50(output_dim):
    return _replace_fc(torchvision.models.resnet50(weights=None), output_dim)


def imagenet_resnet50_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet50(weights="DEFAULT"), output_dim)


def imagenet_resnet18_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet18(weights="DEFAULT"), output_dim)


def imagenet_resnet34_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet34(weights="DEFAULT"), output_dim)


def imagenet_resnet101_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet101(weights="DEFAULT"), output_dim)


def imagenet_resnet152_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet152(weights="DEFAULT"), output_dim)


def imagenet_densenet121_pretrained(output_dim):
    return _densenet_replace_fc(
        torchvision.models.densenet121(weights="DEFAULT"), output_dim
    )


def imagenet_densenet121(output_dim):
    return _densenet_replace_fc(
        torchvision.models.densenet121(weights=None), output_dim
    )


def imagenet_vgg19_pretrained(output_dim):
    model = torchvision.models.vgg19(weights="DEFAULT")
    return _vgg_replace_fc(model, output_dim)


def imagenet_vgg16_pretrained(output_dim):
    model = torchvision.models.vgg16(weights="DEFAULT")
    return _vgg_replace_fc(model, output_dim)


def imagenet_alexnet_pretrained(output_dim):
    model = torchvision.models.alexnet(weights="DEFAULT")
    model.classifier[0].in_features = model.classifier[1].in_features
    return _vgg_replace_fc(model, output_dim)


def _base_resnet18_cifar():
    # much wider than cifar_preresnet20
    model = torchvision.models.resnet18(weights=None)  # load model from torchvision.models without pretrained weights.
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def cifar_resnet18(output_dim):
    model = _base_resnet18_cifar()
    return _replace_fc(model, output_dim)
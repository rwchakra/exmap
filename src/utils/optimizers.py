import torch


def sgd_optimizer_fromparams(params, lr, momentum, weight_decay):
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum,
        weight_decay=weight_decay)
    return optimizer



def sgd_optimizer(model, lr, momentum, weight_decay):
    return sgd_optimizer_fromparams(
        model.parameters(), lr, momentum, weight_decay)


def adamw_optimizer(model, lr, momentum, weight_decay):
    del momentum
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=weight_decay)
    return optimizer


def cosine_lr_scheduler(optimizer, num_steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps)


def constant_lr_scheduler(optimizer, num_steps):
    return None
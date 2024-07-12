import os
import numpy as np
from tqdm import tqdm
import torch
import zennit


class GammaSymmetric(zennit.core.BasicHook):

    def __init__(self, gamma=0.25):
        self.gamma = gamma

        def gradient_map(out_grad, outputs):
            P = outputs[0] + outputs[1]  # positive contributions
            N = outputs[2] + outputs[3]  # negative contributions

            alpha = 1 + gamma * (P >= N.abs()) # gamma for positive contr.
            beta = 1 + gamma * (P < N.abs())  # gamma for negative contr.

            out = P * alpha + N * beta

            factors = [alpha, alpha, beta, beta]

            return [fac * out_grad / 
                zennit.core.stabilize(out) for fac in factors]

        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
            ],
            param_modifiers=[
                lambda param, _: param.clamp(min=0),
                lambda param, name: param.clamp(max=0) if name != 'bias' else torch.zeros_like(param),
                lambda param, _: param.clamp(max=0),
                lambda param, name: param.clamp(min=0) if name != 'bias' else torch.zeros_like(param),
            ],
            output_modifiers=[lambda output: output] * 4,
            gradient_mapper=(lambda out_grad, outputs: gradient_map(out_grad, outputs)),
            reducer=(
                lambda inputs, gradients: (
                        (inputs[0] * gradients[0] + inputs[1] * gradients[1]) +
                        (inputs[2] * gradients[2] + inputs[3] * gradients[3])
                )
            )
        )


def lrp(args, model, loader):
    heatmaps = []
    ys = []
    gs = []
    preds = []
    for batch in tqdm(loader):
        x, y, g, _ = batch
        x = x.cuda()

        # compute the prediction for batch
        with torch.no_grad():
            out = model(x)
        pred = out.argmax(dim=1)

        torch.cuda.empty_cache()

        # compute the heatmaps for batch using prediction
        canonizer = zennit.torchvision.ResNetCanonizer()
        composite = zennit.composites.EpsilonAlpha2Beta1(canonizers=[canonizer])

        # overwrite the default rule for convolutional layers due to numerical instability in implementation
        new_layer_map = []
        for type, rule in composite.layer_map:
            if type == zennit.types.Convolution:
                new_layer_map.append((type, GammaSymmetric(gamma=4)))
            else:
                new_layer_map.append((type, rule))
        composite.layer_map = new_layer_map

        target = torch.eye(2).cuda()[pred]
        with zennit.attribution.Gradient(
            model=model, composite=composite
        ) as attributor:
            # compute the model output and attribution
            _, res = attributor(x, target)
            res.detach_()

        if args.downsize is not None:
            res = torch.nn.functional.interpolate(
                res, size=args.downsize, mode="bilinear"
            )
        res = res.reshape(res.shape[0], -1)
        res = res.cpu().numpy()

        heatmaps += [heat for heat in res]
        preds.append(pred.cpu().numpy())
        ys.append(y.cpu().numpy())
        gs.append(g.cpu().numpy())

    ys = np.concatenate(ys, axis=0)
    gs = np.concatenate(gs, axis=0)
    preds = np.concatenate(preds, axis=0)

    print("Accuracy: ", np.sum(preds == ys) / len(ys))
    # save(args, heatmaps, ys, gs, preds)

    return heatmaps, ys, gs, preds


def feature_extraction(args, model, loader):
    # separate the last layer from the rest of the model
    fc = model.fc
    model.fc = torch.nn.Identity()

    features = []
    preds = []
    ys = []
    gs = []
    for batch in tqdm(loader):
        x, y, g, _ = batch
        x = x.cuda()

        # compute the prediction for batch
        with torch.no_grad():
            feature_extractor = model(x)
            out = fc(feature_extractor)
        pred = out.argmax(dim=1)

        torch.cuda.empty_cache()

        # compute the features for batch using prediction
        features += [feat for feat in feature_extractor.cpu().numpy()]
        preds.append(pred.cpu().numpy()) 
        ys.append(y.cpu().numpy())
        gs.append(g.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    ys = np.concatenate(ys, axis=0)
    gs = np.concatenate(gs, axis=0)
    return features, preds, ys, gs
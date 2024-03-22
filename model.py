import torch

def deep_supervision_loss(preds, label, base_loss_function, deep_layers_weight_factor=1):
    if preds.dim() > label.dim():
        preds = torch.unbind(preds, dim=1)
    else:
        return base_loss_function(preds, label)
    n_preds = len(preds)
    weights = 2**torch.arange(n_preds - 1, -1, -1)
    weights = weights / weights.sum() # for n_preds=3: [4/7 , 2/7, 1/7] each layer is twice as important as it's lower neigbour
    # loss = torch.tensor([weight*base_loss_function(pred, label) for weight, pred in zip(weights, preds)], requires_grad=True).sum()
    out_layer_weight_addition = weights[1:].sum() * (1-deep_layers_weight_factor)
    weights[0] += out_layer_weight_addition
    weights[1:] *= deep_layers_weight_factor
    loss = None
    for weight, pred in zip(weights, preds):
        if loss is None:
            loss = base_loss_function(pred, label)*weight
        else:
            loss += base_loss_function(pred, label)*weight

    if loss < 0:
        raise(ValueError(f"loss should be non negative but got loss={loss}"))

    return loss
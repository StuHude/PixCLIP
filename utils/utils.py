


def get_module_device(module):
    return next(module.parameters()).device if any(p.device for p in module.parameters()) else torch.device('cpu')



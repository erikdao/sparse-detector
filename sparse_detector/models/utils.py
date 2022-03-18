"""
Model's utilities
"""
from tabulate import tabulate


def _get_top_level_module_name(model):
    names = [name for name, _ in model.named_modules()]
    prefixes = [name.split(".")[0] for name in names if name]
    return set(prefixes)


def count_module_params(module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad is False)

    return trainable, non_trainable


def describe_model(model):
    keys = _get_top_level_module_name(model)
    
    data = [("Module", "Trainable", "Non-trainable")]
    trainable_sum, non_trainable_sum = 0, 0

    for key in keys:
        module = getattr(model, key)
        trainable, non_trainable = count_module_params(module)
        data.append((key, trainable, non_trainable))
        trainable_sum += trainable
        non_trainable_sum += non_trainable

    data.append(("", None, None))
    data.append(("Total", trainable_sum, non_trainable_sum))
    print(tabulate(data, headers="firstrow", tablefmt="psql"))

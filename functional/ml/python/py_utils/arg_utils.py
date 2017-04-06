

def parse_kwargs(kwargs, prefix):
    keys = [k for k in kwargs if k.startswith(prefix)]
    new_kwargs = {k.replace(prefix, '', 1): kwargs[k] for k in keys}
    old_kwargs = {k: kwargs[k] for k in kwargs if k not in keys}
    return new_kwargs, old_kwargs

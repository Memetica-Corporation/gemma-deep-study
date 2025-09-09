import sys
import types

# Ensure a minimal wandb stub that sets __spec__ to satisfy importlib checks
if 'wandb' not in sys.modules:
    stub = types.ModuleType('wandb')
    stub.__dict__.update({
        'init': lambda **kwargs: None,
        'watch': lambda *args, **kwargs: None,
        'log': lambda *args, **kwargs: None,
        '__spec__': types.SimpleNamespace(),
        '__path__': [],
        '__file__': __file__,
    })
    sys.modules['wandb'] = stub

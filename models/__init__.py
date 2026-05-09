from .sampling import *

_SAMPLING_EXPORTS = {name for name in globals() if not name.startswith("_")}
_LLADA_EXPORTS = {"LLaDAModelLM", "LLaDAConfig"}
_SDAR_EXPORTS = {"SDARModel", "SDARForCausalLM", "SDARConfig"}
_DREAM_EXPORTS = {"DreamTokenizer", "DreamModel", "DreamConfig"}


def __getattr__(name):
    if name in _LLADA_EXPORTS:
        from . import llada

        value = getattr(llada, name)
    elif name in _SDAR_EXPORTS:
        from . import sdar

        value = getattr(sdar, name)
    elif name in _DREAM_EXPORTS:
        from . import dream

        value = getattr(dream, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


__all__ = sorted(_SAMPLING_EXPORTS | _LLADA_EXPORTS | _SDAR_EXPORTS | _DREAM_EXPORTS)

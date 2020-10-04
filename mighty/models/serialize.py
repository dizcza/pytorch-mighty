from abc import ABC

import torch.nn as nn


class SerializableModule(nn.Module, ABC):
    """
    A serializable module to easily save and restore the attributes, defined
    in `state_attr`.

    Attributes
    ----------
    state_attr : list of str
        A list of module attribute names to be a part of a state dict - the
        result of :func:`SerializableModule.state_dict`.
    """

    state_attr = []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = super().state_dict(destination=destination,
                                         prefix=prefix, keep_vars=keep_vars)
        for attribute in self.state_attr:
            destination[prefix + attribute] = getattr(self, attribute)
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys,
                              error_msgs):
        state_dict_keys = list(state_dict.keys())
        for attribute in self.state_attr:
            key = prefix + attribute
            if key in state_dict_keys:
                setattr(self, attribute, state_dict.pop(key))
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict=state_dict, prefix=prefix,
                                      local_metadata=local_metadata,
                                      strict=strict, missing_keys=missing_keys,
                                      unexpected_keys=unexpected_keys,
                                      error_msgs=error_msgs)

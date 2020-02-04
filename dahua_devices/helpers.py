"""Helpers for Dahua package"""
from collections import Mapping
from functools import singledispatch
from types import SimpleNamespace
from typing import Union, Any, Dict, List, Iterable, Optional

from .exceptions import UpdateRequiredException
from .types import Payload

EncodedData = Union[Dict[str, str], str]
SupportedOutTypes = Union[None, str, int, float, bool]
DecodedData = Union[list, dict, SupportedOutTypes]


def dict_merge(target: dict, source: Mapping):
    target_is_dict = isinstance(target, dict)
    for k, v in source.items():
        if k in target and target_is_dict and isinstance(v, Mapping):
            dict_merge(target[k], source[k])
        else:
            target[k] = source[k]
    return target


def str_value_to_type(value: str) -> SupportedOutTypes:
    if value.isdigit():
        return int(value)
    elif value.count('.') == 1 and all([p.isdigit() for p in value.split('.')]):
        return float(value)
    else:
        lower_value = value.lower()
        if lower_value == 'true':
            return True
        elif lower_value == 'false':
            return False
        elif lower_value == 'null':
            return None
    return value


def type_to_str_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    elif isinstance(value, bool):
        return 'true' if value is True else 'false'
    elif value is None:
        return 'null'
    elif isinstance(value, Iterable):
        return '[' + ','.join([type_to_str_value(a) for a in value]) + ']'
    return str(value)


def _get_encode_prefix(key: Union[int, str], prefix: Optional[str] = None):
    if isinstance(key, int):
        str_idx = '[' + str(key) + ']'
        return prefix + str_idx if prefix else str_idx
    return prefix + '.' + key if prefix else key


def encode_table(data: Payload, prefix: str = None, combined: bool = True) -> EncodedData:
    """Encode Dahua table response"""
    result = {}
    if isinstance(data, Mapping):
        for key, item in data.items():
            if isinstance(key, Iterable) and not isinstance(key, str):
                for in_key in key:
                    sub_prefix = _get_encode_prefix(in_key, prefix)
                    result.update(encode_table(item, sub_prefix, False))
            else:
                sub_prefix = _get_encode_prefix(key, prefix)
                result.update(encode_table(item, sub_prefix, False))

    elif isinstance(data, Iterable) and not isinstance(data, str):
        for idx, item in enumerate(data):
            sub_prefix = _get_encode_prefix(idx, prefix)
            result.update(encode_table(item, sub_prefix, False))

    else:
        if prefix is None:
            raise ValueError('Provided data is not a mapping (%s)' % type(data))
        result[prefix] = type_to_str_value(data)

    return '\n'.join([
        pair[0] + '=' + pair[1]
        for pair in result.items()
    ]) if combined else result


def decode_table(data: Iterable[str], prefix: Optional[str] = None) -> DecodedData:
    """Decode Dahua table response"""
    lines = data.splitlines() if isinstance(data, str) else data
    result = {}
    prefix_len = None
    if prefix:
        if '=' in prefix:
            raise ValueError('Prefix contains an equals ("=") sign character')
        prefix_len = len(prefix)

    for line in lines:
        parts = line.split('=', 1)
        if len(parts) != 2:
            raise ValueError('Line "%s" is not a table entry' % line)

        key, value = parts  # type: str, str
        if prefix:
            if not key.startswith(prefix):
                raise ValueError('Line "%s" is not prefixed with provided prefix "%s"' % (line, prefix))
            key = key[prefix_len:]

        if key == '':
            # prefix eliminated pre-equals part, this should be intentional
            return str_value_to_type(value)

        # split first by brackets because [0].[0]=0 should evaluate to {'':[0]}, not [[0]]
        # also not viable to replace two splits with regex due to >2x increase in performance penalty
        bracket_parts = key.split('[')
        if bracket_parts[0] == '':
            del bracket_parts[0]

        current_object = result
        leaf = None
        for bracket_part in bracket_parts:
            dot_parts = bracket_part.split('.')
            if dot_parts[0] == '':
                del dot_parts[0]
            for part in dot_parts:
                if leaf is not None:
                    current_object = current_object.setdefault(leaf, dict())
                leaf = int(part[:-1]) if ']' in part else part

        current_object[leaf] = str_value_to_type(value)

    return result


def list_as_numbered_dict(data: List[Any]) -> Dict[int, Any]:
    return {index: value for index, value in enumerate(data)}


class DictAttributeWrapper(dict):
    def __init__(self, source_dict: dict):
        super().__init__([(k, v) for k, v in source_dict.items()])
        self.is_current_root = True

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, item, *args, **kwargs):
        value = super().__getitem__(item)
        if isinstance(value, dict):
            result = self.__class__(*args, value, **kwargs)
            result.is_current_root = False
            return result
        return value


class ConfigAccessor(DictAttributeWrapper):
    def __init__(self, device: '_BaseDevice', source_dict: dict):
        super().__init__(source_dict)
        self.device = device

    def __getitem__(self, item, **kwargs):
        try:
            return super().__getitem__(item, self.device)
        except KeyError as e:
            if self.is_current_root:
                raise UpdateRequiredException(self.device, "Device needs to be updated before accessing config key '%s'" % item)
            raise e

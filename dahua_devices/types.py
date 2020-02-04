"""Type definitions used in Dahua project"""
from typing import Dict, Any, Callable, Union, Mapping, Iterable

# Update functions
InfoRetVal = Dict[str, Any]
FnDevice = Callable[[str], InfoRetVal]
FnChannel = Callable[[str, int, InfoRetVal], InfoRetVal]
FnChannelMerge = Callable[[str], InfoRetVal]
InfoParsers = Union[FnDevice, FnChannel, FnChannelMerge]

# Device attributes
DeviceInfo = InfoRetVal
ChannelIndex = int
ChannelsInfo = Dict[ChannelIndex, InfoRetVal]

# Config data
ConfigData = Union[dict, int]
Payload = Union[Mapping, Iterable]

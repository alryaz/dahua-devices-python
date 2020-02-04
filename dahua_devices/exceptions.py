""" Exceptions for Dahua package """
from typing import TYPE_CHECKING, Optional, Any
from urllib.request import Request

if TYPE_CHECKING:
    from .device import _BaseDevice
    from .channel import Channel


class DahuaException(BaseException):
    pass


class DeviceException(DahuaException):
    def __init__(self, device: Optional['_BaseDevice'], *args) -> None:
        super().__init__(*args)
        self.device = device


class ProtocolException(DeviceException):
    default_code = None
    default_reason = None

    def __init__(self, device: '_BaseDevice', request_object: Any = None, code: int = None, reason: str = None, *args):
        self.device = device
        self.code = self.default_code if code is None else code
        self.reason = self.default_reason if reason is None else reason
        self.request_object = request_object

        # @TODO: check whether this call should be at the bottom
        if isinstance(self.request_object, Request):
            super().__init__(device, self.code, self.reason, request_object.full_url)
        else:
            super().__init__(device, self.code, self.reason)

    def __str__(self):
        message = f'Protocol error {self.code}: {self.reason}'
        if isinstance(self.request_object, Request):
            return message + f' ({self.request_object.full_url})'
        return message

    def __int__(self):
        return self.code


class UnauthorizedException(ProtocolException):
    default_code = 401
    default_reason = 'Unauthorized'


class BadRequestException(ProtocolException):
    default_code = 400
    default_reason = 'Bad Request'


class InvalidResponseException(ProtocolException):
    default_code = None
    default_reason = 'Server provided invalid response'


class UpdateRequiredException(DeviceException):
    pass


class ChannelException(DahuaException):
    def __init__(self, channel: Optional['Channel'], *args) -> None:
        super().__init__(*args)
        self.channel = channel


class ChannelMissingException(UpdateRequiredException, ChannelException):
    """Channel is missing from data entries"""
    pass


class ConfigKeyException(ValueError, DahuaException):
    """Provided configuration key is invalid (base class)"""
    def __init__(self, configuration_key: Optional[str], *args):
        super().__init__(*args)
        self.config_key = configuration_key


class ConfigKeyUnsupportedException(ConfigKeyException):
    """Provided configuration key is missing from supported configuration keys dictionary"""
    pass


class ConfigKeyChannelsException(ConfigKeyException):
    """Provided configuration key can be used only in channel management context"""
    pass

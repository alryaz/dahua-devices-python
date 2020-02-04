"""Dahua device specification for Dahua package"""
__all__ = [
    '_BaseDevice',
    'Device',
    'EventsListener'
]

import asyncio
import logging
import threading
import urllib.request
from base64 import b64encode
from socket import timeout as socket_timeout_exception
from typing import Optional, List, Any, Dict, Callable, Union, Tuple, Mapping, Iterable
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode, quote as urlquote

from .channel import Channel
from .const import CGI_EVENTS, CGI_CONFIG, SCRIPT_MAGIC_BOX, SCRIPT_DEV_VIDEO_INPUT
from .exceptions import UnauthorizedException, ProtocolException, UpdateRequiredException, ChannelMissingException, \
    BadRequestException
from .helpers import str_value_to_type, decode_table, dict_merge, encode_table, ConfigAccessor
from .types import ConfigData, Payload

_LOGGER = logging.getLogger(__name__)


class ContextDevice:
    def __init__(self, device):
        self._device = device
        self._config = {}

    def __getattr__(self, item):
        return getattr(self._device, item)

    @property
    def current_transaction(self):
        return self._config

    def set_config(self, config_key: str, payload: dict):
        target = self._config.setdefault(config_key, {})
        dict_merge(target, payload)


class _BaseDevice:
    _configuration_device: Dict = NotImplemented
    _configuration_channel: Dict = NotImplemented
    _aliases: Dict = NotImplemented
    _actions_cache: Dict[str, str] = dict()

    def __init__(self, host: str, port: int, username: str, password: str, use_ssl=False, use_digest_auth=False,
                 channel_number_offset=1, rtsp_port=554) -> None:
        """ Create Dahua device adapter """
        super().__init__()

        # New version
        self._cache: Dict[str, Any] = {}
        self._context = None

        # Device-specific config
        self.channel_number_offset = channel_number_offset
        self.use_ssl = use_ssl
        self.host = host
        self.port = port
        self.rtsp_port = rtsp_port
        self.__username = username
        self.__password = password
        # @TODO: automatic generation when proxy support is added
        self.__auth_basic = 'Basic ' + b64encode(f'{username}:{password}'.encode()).decode('utf-8')
        # @TODO: currently unused
        self.use_digest_auth = use_digest_auth
        self.__auth_digest = None
        if use_digest_auth:
            raise Exception('Digest authentication is not currently supported')

        # Channels (filled upon invoking request for channels-related config update)
        self._channels: Dict[int, Channel] = {}

    def __dir__(self) -> Iterable[str]:
        """Get all methods and attributes as well as dynamic methods"""
        all_attrs = [*super().__dir__()]#, *self.additional_magic_box]
        for attr_name in ['_configuration_device', '_configuration_channel']:
            for aliases in getattr(self, attr_name).values():
                all_attrs.extend([
                    prefix + alias + '_config'
                    for alias in aliases
                    for prefix in ['get_', 'set_']
                ])

        return all_attrs

    def __repr__(self) -> str:
        """String representation of device's current config"""
        return f'<Dahua:Device({self.host}:{self.port})>'

    def __getitem__(self, index: Union[int, str, Tuple[str,int]]) -> Union[Channel, Iterable, Mapping]:
        try:
            if isinstance(index, int):
                return self.get_channel(index)

            if isinstance(index, tuple):
                if len(index) != 2:
                    raise IndexError('Tuple indices must be of length 2 (got: %d)' % len(index))
                return self.get_channel_config(config_key=index[0], index=int(index[1]), from_cache=True)

            elif isinstance(index, str):
                config_key = self._configuration_device.get(index)
                if config_key:
                    return self.get_config_raw(raw_config_key=config_key, from_cache=True)

                if index in self._configuration_channel:
                    return self.get_channel_config(config_key=index, from_cache=True)

                raise IndexError('Mapping not found for configuration key "%s"' % index)

            raise TypeError("Allowed index types are %s (got: %s)" % ([int, str, tuple], type(index)))

        except ValueError as e:
            raise IndexError(*e.args) from None

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name.endswith('_config'):
            is_get = attr_name.startswith('get_')
            if not is_get:
                is_set = attr_name.startswith('set_')
                if not is_set:
                    raise AttributeError("Invalid configuration method '%s'" % attr_name)
            alias = attr_name[4:-7]

            config_key = self._configuration_device.get(alias)
            if config_key is None:
                config_key = self._configuration_channel.get(alias)
                if config_key is None:
                    raise AttributeError("Unregistered configuration method '%s'" % attr_name)
                elif is_get:
                    return lambda *arg, **kwargs: self.get_channel_config(config_key, *arg, **kwargs)
                return lambda *arg, **kwargs: self.set_channel_config(config_key, *arg, **kwargs)
            elif is_get:
                return lambda *arg, **kwargs: self.get_device_config(config_key, *arg, **kwargs)
            return lambda *arg, **kwargs: self.get_device_config(config_key, *arg, **kwargs)

        elif attr_name in self._configuration_device.values() or \
                attr_name in self._configuration_channel.values():
            # special case for 'config' cache realm
            config = self.cache(realm='config')
            _LOGGER.debug('Config: %s' % config)
            if attr_name not in config:
                raise UpdateRequiredException(self, "Update required to access cached config key '%s'" % attr_name)
            return ConfigAccessor(self, config)[attr_name]

        for call_config, aliases in self._aliases.items():
            if attr_name in aliases:
                cache_key = 'action_' + attr_name if attr_name in self._actions_cache else None
                if cache_key and cache_key in self._cache:
                    return lambda *args, **kwargs: self._cache[cache_key]

                if isinstance(call_config, str):
                    call_config = [call_config]
                method = super().__getattribute__(call_config[0])

                alias_params = aliases[attr_name]
                if isinstance(alias_params, str):
                    alias_params = [alias_params]

                method_args = call_config[1:]
                if cache_key:
                    def method_call(*args, **kwargs):
                        if cache_key in self._cache:
                            return self._cache[cache_key]

                        # noinspection PyBroadException
                        try:
                            result = method(*method_args, *alias_params, *args, **kwargs)
                        except:
                            result = None

                        self._cache[cache_key] = result
                        return result

                    return method_call
                return lambda *args, **kwargs: method(*method_args, *alias_params, *args, **kwargs)

        raise AttributeError("Object does not contain attribute '%s'" % attr_name)

    def __enter__(self):
        self._context = ContextDevice(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            c_t = self._context.current_transaction
            request_url = self._generate_url_set_config(c_t)
            response_obj, response_text = self.make_request(request_url)
            self._process_response(response_obj, response_text)

            config_cache = self.cache(realm='config')
            for config_key in c_t:
                if config_key in config_cache:
                    del config_cache[config_key]

    @property
    def channels(self) -> Dict[int, Channel]:
        return self._channels

    def _update_channels(self, from_config: Dict[int, Any]):
        current_channels = self._channels
        for channel_index in from_config:
            if channel_index not in current_channels:
                current_channels[channel_index] = Channel(self, channel_index)

    def cache(self, save_data: Optional[Union[dict, bool]] = None, realm: str = 'default') -> dict:
        """
        Simplistic cache manager for object.
        :param save_data: Data to save to the cache dictionary
        :param realm: Cache realm (for requests)
        :return: Cache dictionary for realm, or root cache dictionary in case of clearing
        """
        if save_data is False:
            # save data is forcefully set to False
            cache = {}
            if realm is None:
                # clear all cache
                self._cache = cache
            else:
                # clear specific realm
                self._cache[realm] = cache
        elif realm is not None:
            # normal data retrieval
            cache = self._cache.setdefault(realm, {})
            if save_data:
                # save provided data
                dict_merge(cache, save_data)
        else:
            # return root cache object
            return self._cache
        return cache

    # HTTP request preparations
    def generate_url(self, path, protocol=None, add_credentials=False, port=None, **kwargs):
        """Generate URL"""
        prefix = (protocol if protocol else 'https' if self.use_ssl else 'http') + '://'
        if add_credentials:
            prefix += urlquote(self.__username, safe='') + ':' + urlquote(self.__password, safe='') + '@'
        base_url = prefix + self.host + ':' + str(self.port if port is None else port) + path

        return base_url + '?' + urlencode(kwargs, safe=',') if kwargs else base_url

    def _generate_request_headers(self, headers: Optional[dict] = None) -> dict:
        """Generate headers for requests (including authorization)"""
        request_headers = {}
        if self.__auth_basic:
            # @TODO: this clause is here for future proxy support
            request_headers['Authorization'] = self.__auth_basic
        if headers:
            request_headers.update(headers)
        elif headers:
            request_headers = headers
        return request_headers

    def _process_response(self, response_obj: Any, response_text: Optional[str], prefix: Optional[str] = None,
                          cache_realm: Optional[str] = 'default', cache_key: Optional[str] = None
                          ) -> Optional[Union[bool, list, dict]]:
        """Fetched configuration response helper (used with"""
        if not response_text:
            return None

        len_response = len(response_text)
        if len_response in [2, 5]:
            response_lower = response_text.lower()
            if response_lower == 'error':
                raise BadRequestException(self, response_obj)
            elif response_lower == 'ok':
                return True

        try:
            data = decode_table(response_text, prefix=prefix)
            if cache_realm is not None:
                self.cache(data if cache_key is None else {cache_key: data}, cache_realm)
            return data

        except ValueError:
            raise ProtocolException(self)

    def _generate_url_set_config(self, payload: Payload, config_key: Optional[str] = None) -> str:
        """
        Generate payload URL for setting configuration data on device.
        :param config_key: Configuration key (and channel index, if provided)
        :param payload:
        :param channel_index: (optional) Channel index, or list of indices

        :return: Parameter setting URL
        :rtype: str
        """

        table_values = encode_table(payload, prefix=config_key, combined=False)

        return self.generate_url(
            path='/cgi-bin/configManager.cgi',
            action='setConfig',
            **table_values
        )

    def get_channel(self, index: int, is_number: bool = False) -> Optional[Channel]:
        """Get channel object by its index (or number)"""
        channel = self._channels.get(index + self.channel_number_offset if is_number else index)
        if channel is None:
            raise ChannelMissingException(self)
        return channel

    def make_request(self, url, headers=None, decode=True):
        print('this', url)
        request = urllib.request.Request(
            url=url,
            headers=self._generate_request_headers(headers),
            method='GET'
        )
        response_obj = None
        try:
            _LOGGER.debug('Making request to `%s`' % url)
            with urllib.request.urlopen(request) as response_obj:
                if not decode:
                    _LOGGER.debug('Returning request to `%s`' % url)
                    return response_obj

                response_text = response_obj.read().decode('utf-8')
                _LOGGER.debug('Finished request to `%s`' % url)

                return response_obj, response_text.replace('\r\n', '\n').strip()

        except HTTPError as e:
            if e.code == 401:
                raise UnauthorizedException(self, response_obj) from None
            raise ProtocolException(self, response_obj, e.code, e.reason) from None

        except URLError as e:
            raise ProtocolException(self, response_obj, e.errno, e.reason) from None

    def fetch_config(self, raw_config_key: str, update_cache: bool = True) -> Optional[ConfigData]:
        """
        Fetch data for given configuration key from the device.
        :param raw_config_key: Configuration key
        :param update_cache: Update cache after request

        :returns: Data related to configuration key
        :rtype: dict, list, NoneType
        """
        request_url = self.generate_url(path=CGI_CONFIG, action='getConfig', name=raw_config_key)
        response_obj, response_text = self.make_request(request_url)
        result = self._process_response(response_obj, response_text, 'table.' + raw_config_key, 'config', raw_config_key)
        if update_cache and isinstance(result, dict):
            self.cache(result, 'config')
        return result

    def get_config_raw(self, raw_config_key: str, from_cache: bool = True) -> Optional[ConfigData]:
        """
        Get configuration data from cache, or fetch it in case it is missing.
        :param raw_config_key: Raw configuration key
        :param from_cache: Check cache for data

        :returns: Data related to configuration key, if not empty
        :rtype: dict, list, NoneType
        """
        if not from_cache:
            return self.fetch_config(raw_config_key)

        config = self.cache(realm='config').get(raw_config_key)
        if config is None:
            return self.fetch_config(raw_config_key)

        return config

    def get_device_config(self, config_key: str, from_cache: bool = True) -> Optional[ConfigData]:
        """
        Get configuration data for device.

        This function is a proxy for `get_config_raw`. It checks configuration key against
        the dictionary of supported device configuration keys, and maps it to a raw config
        key for fetching from cache or from device directly.

        :param config_key: Configuration key
        :param from_cache: Check cache for data

        :returns: Data related to configuration key, if not empty
        :rtype: dict, list, NoneType
        """
        raw_config_key = self._configuration_device.get(config_key)
        if raw_config_key is None:
            raise ValueError('Config key "%s" is not mapped to any raw device config keys.' % config_key)

        return self.get_config_raw(raw_config_key, from_cache)

    def get_channel_config(self, config_key: str, index: Optional[int] = None, from_cache: bool = True) -> Optional[ConfigData]:
        """
        Get configuration data for channel(s).

        This function is a proxy for `get_config_raw`. It checks configuration key against
        the dictionary of supported channel configuration keys, and maps it to a raw config
        key for fetching from cache or from device directly.

        :param config_key: Configuration key
        :param index: (optional) Channel index
        :param from_cache: Check cache for data

        :returns: Data related to configuration key for given channel, if not empty
        :rtype: dict, list, NoneType
        """
        raw_config_key = self._configuration_channel.get(config_key)
        if raw_config_key is None:
            raise ValueError('Config key "%s" is not mapped to any raw channel config keys.' % config_key)

        needs_update = not from_cache or raw_config_key not in self.cache(realm='config')
        config = self.get_config_raw(raw_config_key, from_cache)

        if config is None:
            return config

        if needs_update:
            self._update_channels(config)

        return config if index is None else config.get(index)

    def set_config(self, config_key: str, payload: Payload, reset_cache: bool = True) -> Optional[bool]:
        """
        Send configuration data to device and wait for response.
        :param config_key: Configuration key
        :param payload: Updated data related to configuration key
        :param reset_cache: Reset config key cache after request

        :returns: Payload delivery response contains 'OK', or nothing while in context
        :rtype: bool, None
        """
        request_url = self._generate_url_set_config(payload, config_key)
        response_obj, response_text = self.make_request(request_url)
        result = self._process_response(response_obj, response_text)

        res_is_true = result is True
        if reset_cache and res_is_true:
            cache = self.cache(realm='config')
            if config_key in cache:
                del cache[config_key]

        return res_is_true

    def set_channel_config(self, config_key: str, index: Union[int, Iterable[int]], payload: Payload,
                           reset_cache: bool = True) -> Optional[bool]:
        """
        Send configuration data for given channel to device and wait for response.
        :param config_key: Configuration key
        :param index: Channel index, or list of channel indices
        :param payload: Updated data related to configuration key for given channel
        :param reset_cache: Reset config key cache after request

        :returns: Payload delivery response contains 'OK'
        :rtype: bool
        """
        if config_key not in self._configuration_channel.values():
            raise ValueError('Config key "%s" cannot be used in channel context' % config_key)

        combo_index = (index,) if isinstance(index, int) else (*index,)
        result = self.set_config(config_key, {combo_index: payload}, False)

        res_is_true = result is True
        if res_is_true and reset_cache:
            cache = self.cache(realm='config')
            for index in combo_index:
                if index in cache:
                    del cache[index]

        return res_is_true

class Device(_BaseDevice):
    _configuration_device = {
        "auto_maintain":    "AutoMaintain",
        #"max_extra_stream": "MaxExtraStream",
        "video_standard":   "VideoStandard",
        "network":          "Network",
        "pppoe":            "PPPoE",        # 5.3.1, 5.3.2
        "ddns":             "DDNS",         # 5.4.1, 5.4.2
        "email":            "Email",        # 5.5.1, 5.5.2
        "wlan":             "WLan",         # 5.6.1, 5.6.2 (ut)
        "upnp":             "UPnP",         # 5.7.1, 5.7.2
        "ntp":              "NTP",          # 5.8.1, 5.8.2
        "rtsp":             "RTSP",         # 5.9.1, 5.9.2
        "telnet":           "Telnet",       # 5.10.1, 5.10.2
        "alarm_server":     "AlarmServer",
        "alarm":            "Alarm",
        "alarm_out":        "AlarmOut",
        "general":          "General",
        "locales":          "Locales",
        "language":         "Language",
        "access_filter":    "AccessFilter",
        "flashlight":       "FlashLight",   # 4.12.1, 4.12.2 (ut)
        "video_output":     "VideoOut",     # 4.11.1, 4.11.2
    }
    _configuration_channel = {
        "video_color":      "VideoColor",       # 4.2.1, 4.2.2
        "video_input":      "VideoInOptions",   # 4.3.2, 4.3.3
        "encoding":         "Encode",
        "channel_title":    "ChannelTitle",
        "video_widget":     "VideoWidget",
        "motion_detect":    "MotionDetect",
        "blind_detect":     "BlindDetect",
        "loss_detect":      "LossDetect",
        "ptz":              "Ptz",
        "record":           "Record",
        "record_mode":      "RecordMode",
        "snap":             "Snap",
    }
    _aliases = {
        ("_param_action", "/cgi-bin/wlan.cgi"): {
            "scan_wlan_devices":  ({"SSID": ((str, int,),None)},), # 5.6.3 (ut)
        },
        ("_param_action", "/cgi-bin/devAudioInput.cgi"): {
            "get_audio_input_count": ("getCollect", "result"), # 11.4.1 (ut)
        },
        ("_param_action", "/cgi-bin/devAudioOutput.cgi"): {
            "get_audio_output_count": ("getCollect", "result"),  # 11.5.1 (ut)
        },
        ("_param_action", "/cgi-bin/netApp.cgi"): {
            "get_network_interfaces":           ("getInterfaces", "netInterface"),  # 5.1.1
            "get_upnp_status":                  ("getUPnPStatus", "status"),        # 5.7.3
        },
        ("_param_action", SCRIPT_DEV_VIDEO_INPUT): {
            "get_video_input_count":            ("getCollect", "result"),               # 4.10.1 (ut)
            "get_video_input_capabilities":     ({"action": "getCaps",
                                                  "channel": (int,)}, None),            # 4.3.1 (ut)
            "adjust_focus":                     ({"action": "adjustFocus",              #
                                                  "focus": ((float, int), -1),          #
                                                  "zoom": ((float, int), -1)}, None),   #
            "adjust_focus_continuously":        ({"action": "adjustFocusContinuously",  #
                                                  "focus": ((float, int), -1),
                                                  "zoom": ((float, int), -1)}, None),
            "auto_focus":                       ("autoFocus", False),
            "get_focus_status":                 ("getFocusStatus", "status")
        },
        ("_param_action", SCRIPT_MAGIC_BOX): {
            "reboot":                           ("reboot", False),
            "shutdown":                         ("shutdown", False),
            "get_serial_number":                ("getSerialNo", "sn"),
            "get_device_type":                  ("getDeviceType", "type"),
            "get_hardware_version":             ("getHardwareVersion", "version"),
            "get_software_version":             ("getSoftwareVersion", True),
            "get_machine_name":                 ("getMachineName", "name"),
            "get_system_info":                  ("getSystemInfo", None),
            "get_vendor":                       ("getVendor", "vendor"),
            "get_max_extra_stream":             ({"action": "getProductDefinition",
                                                  "name": "MaxExtraStream"}, ("table", "MaxExtraStream")),
            "get_language_capabilities":        ("getLanguageCaps", lambda result: (result['Languages'].split(',')
                                                                                    if result and 'Languages' in result
                                                                                    else None))
        }
    }
    _actions_cache = ["get_serial_number", "get_software_version", "get_device_type", "get_vendor"]

    def _param_action(self, script_name: str, params: Union[str, Mapping[str, Union[type, str]]],
                      access_key_return: Optional[Union[str, Tuple[Union[int, str]], Callable]] = None, **kwargs):
        """
        Helper method to call parametrized action requests via dynamic accessors.
        :param script_name: Script path on server
        :param params: "action" request parameter value, or dict with request parameters
        :param access_key_return: Return value by access key, or return nothing at all (if False)
        :return: Data relevant to requested dynamic method
        :rtype: dict, None
        """
        if params is None:
            url = self.generate_url(script_name)
        elif isinstance(params, str):
            url = self.generate_url(script_name, action=params)
        else:
            request_params = {}
            for param, value in params.items():
                if isinstance(value, tuple):
                    if param in kwargs:
                        param_arg_value = kwargs[param]
                        if not isinstance(param_arg_value, value[0]):
                            raise TypeError('Parameter "%s" should be of type(s): %s (got %s)'
                                            % (param, value[0], type(param_arg_value)))
                        value = param_arg_value
                    else:
                        if len(value) < 2:
                            # no default value error
                            ValueError('Expecting parameter "%s" of type(s) %s' % (param, value[0]))
                            continue
                        value = value[1]

                if value is not None:
                    # don't set empty parameters (may come from default values or arguments themselves)
                    request_params[param] = str(value)

            url = self.generate_url(script_name, **request_params)

        response_obj, response_text = self.make_request(url)
        result = self._process_response(response_obj, response_text)

        if access_key_return is not False:
            if callable(access_key_return):
                return access_key_return(result)

            if result is not None and access_key_return:
                if isinstance(access_key_return, str):
                    access_key_return = [access_key_return]
                current_object = result
                for key in access_key_return:
                    current_object = current_object[key]
                return current_object
            return result

    @property
    def serial_number(self):
        if 'action_get_serial_number' in self._cache:
            return self._cache['action_get_serial_number']

        return self.get_serial_number()

class EventsListener(threading.Thread):
    def __init__(self,
                 device: _BaseDevice,
                 monitored_events: List[str] = None,
                 alarm_channel: int = 1):
        super().__init__()

        self._callbacks = set()
        self.stopped = threading.Event()
        if monitored_events is None:
            monitored_events = ['All']
        self._monitored_events = monitored_events
        self._device = device
        self._alarm_channel = alarm_channel

    @property
    def device(self) -> _BaseDevice:
        return self._device

    @property
    def monitored_events(self) -> List[str]:
        return self._monitored_events

    @property
    def subscribe_url(self) -> str:
        return self._device.generate_url(
            path=CGI_EVENTS,
            action='attach',
            channel=self._alarm_channel,
            codes='[' + (','.join(self._monitored_events)) + ']',
        )

    def add_event_callback(self, callback: Callable[[_BaseDevice, Dict[str, Any]], Any]) -> None:
        if not callable(callback):
            raise TypeError
        self._callbacks.add(callback)

    def remove_event_callback(self, callback: Callable[[_BaseDevice, Dict[str, Any]], Any]) -> None:
        if not callable(callback):
            raise TypeError
        self._callbacks.remove(callback)

    def listen_events_sync(self) -> bool:
        events_url = self.subscribe_url
        req = self._device._generate_request(events_url)

        try:
            _LOGGER.debug('Started listening `%s` for events...' % events_url)
            with urllib.request.urlopen(req, timeout=30) as response:
                buffer = b''
                last_char = b''
                while True:
                    this_char = response.read(1)
                    if last_char == b'\r' and this_char == b'\n':
                        line = buffer[:-1].decode('utf-8')
                        if line.startswith('Code'):
                            split_parts = [part.split('=', 1) for part in line.split(';')]
                            parts = {
                                part[0].lower(): str_value_to_type(part[1])
                                for part in split_parts
                            }

                            if 'index' in parts:
                                parts['channel'] = self.device.get_channel(parts['index'], is_number=False)

                            for callback in self._callbacks:
                                callback(self, parts)

                        buffer = b''
                    else:
                        buffer += this_char

                    last_char = this_char
        except URLError as e:
            _LOGGER.error('HTTP error occured while listening for %s:%d: %s' % (self._device.host,
                                                                                self._device.port,
                                                                                e.reason))
            return False
        except socket_timeout_exception:
            _LOGGER.debug('Socket timeout during event listening, assumed as graceful')
            return True
        except OSError as e:
            _LOGGER.error('Low-level error `%s` while listening for %s:%d: %s' % (type(e),
                                                                                  self._device.host,
                                                                                  self._device.port,
                                                                                  str(e)))
            return False
        except Exception as e:
            _LOGGER.debug('Exception `%s` during event listening, assumed as graceful: %s' % (type(e), str(e)))
            return True  # in all other cases, listening failed gracefully

    def run(self):
        """Fetch events"""
        while not self.stopped.isSet():
            # Sleeps to ease load on processor

            status = self.listen_events_sync()
            if not status:
                _LOGGER.critical('Events listener failed with unrecovable error (see above)')
                self.stopped.set()

            # loop = asyncio.get_event_loop()
            # future = asyncio.ensure_future(self.listen_events())
            # loop.run_until_complete(future)

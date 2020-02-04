__all__ = [
    'Channel'
]

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Union, Optional, TypeVar

if TYPE_CHECKING:
    from .device import _BaseDevice

ChannelDevice = TypeVar('ChannelDevice', bound='_BaseDevice')
class Channel:
    def __init__(self, device: ChannelDevice, index: int) -> None:
        if device is None:
            raise ValueError('Cannot create channel without parent device')
        self._device = device
        self._index: int = index

    def __getitem__(self, index: str) -> Any:
        """Access channel info via square bracket accessor"""
        try:
            return self._device.get_channel_config(
                config_key=index,
                index=self._index,
                from_cache=True
            )
        except ValueError as e:
            raise IndexError(*e.args) from None

    def __int__(self) -> int:
        return self._index

    def __repr__(self) -> str:
        return '<Dahua:Channel(' + repr(self._device) + ', ' + str(self._index) + ')>'

    @property
    def device(self) -> ChannelDevice:
        return self._device

    @property
    def index(self) -> int:
        return self._index

    @property
    def number(self) -> int:
        return self._index + self._device.channel_number_offset

    @property
    def name(self) -> str:
        return self['channel_title']['Name']

    @property
    def ptz_supported(self) -> bool:
        ptz_config = self['ptz']
        return ptz_config and (
                ptz_config.get('PTZType') == 1
                or ptz_config['ProtocolName'] != 'NONE'
        )

    @property
    def motion_detect_enabled(self) -> bool:
        motion_detect_config = self['motion_detect']
        return motion_detect_config and bool(motion_detect_config.get('Enable', False))

    def get_stream_url(self, stream_type: int = 0, start_time: Optional[datetime] = None,
                       end_time: Optional[Union[datetime, timedelta]] = None) -> str:
        args = {
            'channel': self.number,
            'subtype': stream_type,
        }

        if start_time:
            if end_time is None:
                end_time = datetime.now(tz=start_time.tzinfo)
            elif isinstance(end_time, timedelta):
                end_time = start_time + end_time

            arg_format = '%Y_%m_%d_%H_%M_%S'
            args.update({
                'starttime': start_time.strftime(arg_format),
                'endtime': end_time.strftime(arg_format),
            })
            action = 'playback'
        else:
            action = 'realmonitor'

        return self._device.generate_url(
            path='/cam/' + action,
            port=self.device.rtsp_port,
            protocol='rtsp',
            add_credentials=True,
            **args
        )

    @property
    def stream_url(self):
        """Retrieve primary stream URL."""
        return self.get_stream_url(stream_type=0)

    @property
    def alt_stream_url(self):
        """Retrieve alternative stream URL."""
        return self.get_stream_url(stream_type=1)

    def get_snapshot(self):
        raise NotImplementedError

    async def async_get_snapshot(self):
        raise NotImplementedError

    @property
    def snapshot_url(self):
        return self._device.generate_url(
            path='/cgi-bin/snapshot.cgi',
            channel=self._index
        )

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class QuoteData(_message.Message):
    __slots__ = ("timestamp", "symbol", "bid_price", "ask_price", "bid_size", "ask_size")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    BID_PRICE_FIELD_NUMBER: _ClassVar[int]
    ASK_PRICE_FIELD_NUMBER: _ClassVar[int]
    BID_SIZE_FIELD_NUMBER: _ClassVar[int]
    ASK_SIZE_FIELD_NUMBER: _ClassVar[int]
    timestamp: str
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    def __init__(self, timestamp: _Optional[str] = ..., symbol: _Optional[str] = ..., bid_price: _Optional[float] = ..., ask_price: _Optional[float] = ..., bid_size: _Optional[int] = ..., ask_size: _Optional[int] = ...) -> None: ...

class TradeData(_message.Message):
    __slots__ = ("timestamp", "symbol", "price", "volume", "condition")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SYMBOL_FIELD_NUMBER: _ClassVar[int]
    PRICE_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    timestamp: str
    symbol: str
    price: float
    volume: int
    condition: str
    def __init__(self, timestamp: _Optional[str] = ..., symbol: _Optional[str] = ..., price: _Optional[float] = ..., volume: _Optional[int] = ..., condition: _Optional[str] = ...) -> None: ...

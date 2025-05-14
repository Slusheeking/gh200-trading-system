"""
Models and Enums for Enhanced Fast Exit Strategy
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional

# --- Enums ---

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderClass(str, Enum):
    SIMPLE = "simple"
    BRACKET = "bracket"
    OCO = "oco"
    OTO = "oto"

class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    GTX = "gtx"

class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    PENDING_NEW = "pending_new"
    PENDING_CANCEL = "pending_cancel"

# --- Exceptions ---

class ConfigError(Exception):
    pass

class ValidationError(Exception):
    pass

class ExecutionError(Exception):
    pass

class SharedMemoryError(Exception):
    pass

# --- Data Models ---

@dataclass
class Order:
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: float = 0.0
    stop_price: float = 0.0
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: str = ""
    order_class: OrderClass = OrderClass.SIMPLE
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0
    trail_percent: float = 0.0
    extended_hours: bool = False

    def __post_init__(self):
        if not self.client_order_id:
            self.client_order_id = f"{self.side.value}_{self.symbol}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        if not self.symbol:
            raise ValidationError("Symbol is required")
        if self.quantity <= 0:
            raise ValidationError("Quantity must be positive")
        if self.type == OrderType.LIMIT and self.limit_price <= 0:
            raise ValidationError("Limit price must be positive for limit orders")
        if self.type == OrderType.STOP and self.stop_price <= 0:
            raise ValidationError("Stop price must be positive for stop orders")
        if self.type == OrderType.STOP_LIMIT:
            if self.stop_price <= 0:
                raise ValidationError("Stop price must be positive for stop-limit orders")
            if self.limit_price <= 0:
                raise ValidationError("Limit price must be positive for stop-limit orders")
        if self.order_class == OrderClass.BRACKET:
            if self.take_profit_price <= 0 and self.stop_loss_price <= 0:
                raise ValidationError("At least one of take_profit_price or stop_loss_price must be set for bracket orders")
            if self.take_profit_price > 0:
                if self.side == OrderSide.BUY and self.type == OrderType.LIMIT and self.take_profit_price <= self.limit_price:
                    raise ValidationError("Take profit price must be higher than limit price for buy orders")
                elif self.side == OrderSide.SELL and self.type == OrderType.LIMIT and self.take_profit_price >= self.limit_price:
                    raise ValidationError("Take profit price must be lower than limit price for sell orders")
            if self.stop_loss_price > 0:
                if self.side == OrderSide.BUY and self.type == OrderType.LIMIT and self.stop_loss_price >= self.limit_price:
                    raise ValidationError("Stop loss price must be lower than limit price for buy orders")
                elif self.side == OrderSide.SELL and self.type == OrderType.LIMIT and self.stop_loss_price <= self.limit_price:
                    raise ValidationError("Stop loss price must be higher than limit price for sell orders")

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.type.value,
            "qty": str(self.quantity),
            "time_in_force": self.time_in_force.value,
        }
        if self.limit_price > 0:
            result["limit_price"] = str(round(self.limit_price, 4))
        if self.stop_price > 0:
            result["stop_price"] = str(round(self.stop_price, 4))
        if self.client_order_id:
            result["client_order_id"] = self.client_order_id
        if self.extended_hours:
            result["extended_hours"] = True
        if self.order_class == OrderClass.BRACKET:
            result["order_class"] = "bracket"
            if self.take_profit_price > 0:
                result["take_profit"] = {"limit_price": str(round(self.take_profit_price, 4))}
            if self.stop_loss_price > 0:
                stop_loss = {"stop_price": str(round(self.stop_loss_price, 4))}
                if self.trail_percent > 0:
                    stop_loss["trail_percent"] = str(self.trail_percent)
                result["stop_loss"] = stop_loss
        elif self.order_class == OrderClass.OCO:
            result["order_class"] = "oco"
            if self.take_profit_price > 0:
                result["take_profit"] = {"limit_price": str(round(self.take_profit_price, 4))}
            if self.stop_loss_price > 0:
                result["stop_loss"] = {"stop_price": str(round(self.stop_loss_price, 4))}
        return result

@dataclass
class OrderResponse:
    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    status_message: str = ""
    submission_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    related_orders: List[str] = field(default_factory=list)

    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'OrderResponse':
        try:
            status_str = response.get("status", "new").lower()
            try:
                status = OrderStatus(status_str)
            except ValueError:
                status = OrderStatus.NEW
            order_type_str = response.get("type", "market").lower()
            try:
                order_type = OrderType(order_type_str)
            except ValueError:
                order_type = OrderType.MARKET
            side_str = response.get("side", "buy").lower()
            try:
                side = OrderSide(side_str)
            except ValueError:
                side = OrderSide.BUY
            related_orders = []
            if "legs" in response:
                related_orders = [leg.get("id", "") for leg in response.get("legs", [])]
            return cls(
                order_id=response.get("id", ""),
                client_order_id=response.get("client_order_id", ""),
                symbol=response.get("symbol", ""),
                status=status,
                filled_quantity=float(response.get("filled_qty", "0") or "0"),
                filled_price=float(response.get("filled_avg_price", "0") or "0"),
                status_message=response.get("status_message", ""),
                order_type=order_type,
                side=side,
                submission_time=time.time(),
                last_update_time=time.time(),
                related_orders=related_orders
            )
        except Exception as e:
            return cls(
                client_order_id=response.get("client_order_id", ""),
                symbol=response.get("symbol", ""),
                status=OrderStatus.REJECTED,
                status_message=f"Failed to parse API response: {str(e)}",
                submission_time=time.time(),
                last_update_time=time.time()
            )

@dataclass
class Position:
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    current_price: float = 0.0
    entry_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    highest_price: float = 0.0
    lowest_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    bracket_order_ids: List[str] = field(default_factory=list)
    partial_exits: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = max(self.entry_price, self.current_price)
        if self.lowest_price == 0.0:
            self.lowest_price = min(self.entry_price, self.current_price) if self.current_price > 0 else self.entry_price
        self._update_pnl()

    def update(self, current_price: float) -> None:
        if current_price <= 0:
            return
        self.current_price = current_price
        self.last_update_time = time.time()
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
        self._update_pnl()

    def _update_pnl(self) -> None:
        if self.current_price <= 0 or self.entry_price <= 0:
            return
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_percent = ((self.current_price / self.entry_price) - 1.0) * 100.0
        else:
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.quantity
            self.unrealized_pnl_percent = ((self.entry_price / self.current_price) - 1.0) * 100.0

    def get_duration_seconds(self) -> float:
        return self.last_update_time - self.entry_time

    def get_max_drawdown_percent(self) -> float:
        if self.side == OrderSide.BUY:
            if self.highest_price <= 0:
                return 0.0
            return ((self.highest_price - self.lowest_price) / self.highest_price) * 100.0
        else:
            if self.lowest_price <= 0:
                return 0.0
            return ((self.highest_price - self.lowest_price) / self.lowest_price) * 100.0

    def record_partial_exit(self, exit_price: float, exit_quantity: float, pnl: float) -> None:
        self.partial_exits.append({
            "exit_price": exit_price,
            "exit_quantity": exit_quantity,
            "exit_time": time.time(),
            "pnl": pnl,
            "pnl_percent": (((exit_price / self.entry_price) - 1.0) * 100.0) if self.side == OrderSide.BUY else (((self.entry_price / exit_price) - 1.0) * 100.0)
        })
        self.quantity -= exit_quantity
        self.realized_pnl += pnl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "current_price": self.current_price,
            "entry_time": self.entry_time,
            "last_update_time": self.last_update_time,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_percent": self.unrealized_pnl_percent,
            "realized_pnl": self.realized_pnl,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_id": self.position_id,
            "metadata": self.metadata,
            "bracket_order_ids": self.bracket_order_ids,
            "partial_exits": self.partial_exits,
            "duration_seconds": self.get_duration_seconds(),
            "max_drawdown_percent": self.get_max_drawdown_percent()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        try:
            side_str = data.get("side", "buy").lower()
            side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL
            return cls(
                symbol=data.get("symbol", ""),
                side=side,
                entry_price=float(data.get("entry_price", 0.0)),
                quantity=float(data.get("quantity", 0.0)),
                current_price=float(data.get("current_price", 0.0)),
                entry_time=float(data.get("entry_time", time.time())),
                last_update_time=float(data.get("last_update_time", time.time())),
                highest_price=float(data.get("highest_price", 0.0)),
                lowest_price=float(data.get("lowest_price", 0.0)),
                unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
                unrealized_pnl_percent=float(data.get("unrealized_pnl_percent", 0.0)),
                realized_pnl=float(data.get("realized_pnl", 0.0)),
                stop_loss=float(data.get("stop_loss", 0.0)),
                take_profit=float(data.get("take_profit", 0.0)),
                position_id=data.get("position_id", str(uuid.uuid4())),
                metadata=data.get("metadata", {}),
                bracket_order_ids=data.get("bracket_order_ids", []),
                partial_exits=data.get("partial_exits", [])
            )
        except Exception as e:
            raise ValueError(f"Failed to create Position from dictionary: {str(e)}")

@dataclass
class Signal:
    symbol: str
    type: str
    direction: str
    price: float
    position_size: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0
    timestamp: int = 0
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiration: Optional[int] = None
    is_validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = int(time.time() * 1_000_000_000)
        if self.indicators is None:
            self.indicators = {}
        if self.metadata is None:
            self.metadata = {}
        if self.type == "EXIT" and "exit_reason" not in self.indicators:
            self.indicators["exit_reason"] = "unspecified"
        self.validate()

    def validate(self) -> bool:
        self.validation_errors = []
        if not self.symbol:
            self.validation_errors.append("Symbol is required")
        if not self.type:
            self.validation_errors.append("Signal type is required")
        elif self.type not in ["ENTRY", "EXIT", "ADJUST"]:
            self.validation_errors.append(f"Invalid signal type: {self.type}")
        if not self.direction:
            self.validation_errors.append("Signal direction is required")
        elif self.direction not in ["BUY", "SELL"]:
            self.validation_errors.append(f"Invalid signal direction: {self.direction}")
        if self.price <= 0:
            self.validation_errors.append("Price must be positive")
        if self.position_size <= 0:
            self.validation_errors.append("Position size must be positive")
        if self.type == "ENTRY":
            if self.direction == "BUY":
                if self.stop_loss > 0 and self.stop_loss >= self.price:
                    self.validation_errors.append("Stop loss must be lower than price for buy signals")
                if self.take_profit > 0 and self.take_profit <= self.price:
                    self.validation_errors.append("Take profit must be higher than price for buy signals")
            else:
                if self.stop_loss > 0 and self.stop_loss <= self.price:
                    self.validation_errors.append("Stop loss must be higher than price for sell signals")
                if self.take_profit > 0 and self.take_profit >= self.price:
                    self.validation_errors.append("Take profit must be lower than price for sell signals")
        if self.expiration is not None and self.expiration < int(time.time() * 1_000_000_000):
            self.validation_errors.append("Signal has expired")
        self.is_validated = len(self.validation_errors) == 0
        return self.is_validated

    def is_valid(self) -> bool:
        if not self.is_validated:
            self.validate()
        return self.is_validated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "type": self.type,
            "direction": self.direction,
            "price": self.price,
            "position_size": self.position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "signal_id": self.signal_id,
            "indicators": self.indicators,
            "metadata": self.metadata,
            "expiration": self.expiration,
            "is_validated": self.is_validated,
            "validation_errors": self.validation_errors
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        return cls(
            symbol=data.get("symbol", ""),
            type=data.get("type", ""),
            direction=data.get("direction", ""),
            price=data.get("price", 0.0),
            position_size=data.get("position_size", 0.0),
            stop_loss=data.get("stop_loss", 0.0),
            take_profit=data.get("take_profit", 0.0),
            confidence=data.get("confidence", 0.0),
            timestamp=data.get("timestamp", 0),
            signal_id=data.get("signal_id", str(uuid.uuid4())),
            indicators=data.get("indicators", {}),
            metadata=data.get("metadata", {}),
            expiration=data.get("expiration", None),
            is_validated=data.get("is_validated", False),
            validation_errors=data.get("validation_errors", [])
        )
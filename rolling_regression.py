import asyncio
import itertools
import time
import numpy as np
from collections import deque
from typing import List
from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side

LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS

HEDGE_BOUNDARY = 10
HEDGE_TIME_LIMIT = 58
AVG_WINDOW = 10

MAKER_FEE = 0.0001
TAKER_FEE = 0.0002

HEDGE_RATIO_LOOKBACK = 10
CONSERVATIVE_SPREAD = 0.002
NORMAL_SPREAD = 0.001

def throttler(func):
    max_calls = 47
    interval = 1
    call_times = deque(maxlen=max_calls)
    def wrapper(*args, **kwargs):
        now = time.monotonic()
        if len(call_times) == max_calls and now - call_times[0] < interval:
            return None
        call_times.append(now)
        return func(*args, **kwargs)
    return wrapper

class AutoTrader(BaseAutoTrader):
    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.position = 0
        self.hedge_balance = 0
        self.hedge_time = None
        self.bidq = deque()
        self.askq = deque()
        self.event_loop.call_later(50, self.check_hedge)
        self.etf_prices = deque(maxlen=AVG_WINDOW)
        self.fut_prices = deque(maxlen=AVG_WINDOW)

    @throttler
    def send_throttled(self, client_order_id: int, side: Side, price: int, volume: int):
        if side == Side.BUY:
            if len(self.bidq) >= 4:
                return
            self.bidq.append(client_order_id)
            self.bids.add(client_order_id)
        else:
            if len(self.askq) >= 4:
                return
            self.askq.append(client_order_id)
            self.asks.add(client_order_id)
        self.send_insert_order(client_order_id, side, price, volume, Lifespan.GOOD_FOR_DAY)
    
    def calculate_hedge_ratio(self):
        if len(self.etf_prices) < HEDGE_RATIO_LOOKBACK or len(self.fut_prices) < HEDGE_RATIO_LOOKBACK:
            return None

        etf_prices_array = np.array(self.etf_prices)
        fut_prices_array = np.array(self.fut_prices)
        hedge_ratio = np.polyfit(etf_prices_array, fut_prices_array, deg=1)[0]
        return hedge_ratio

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        if bid_prices[0] <= 0 and ask_prices[0] <= 0:
            return
        
        if instrument == Instrument.ETF:
            self.etf_prices.append((bid_prices[0] + ask_prices[0]) // 2)
        elif instrument == Instrument.FUTURE:
            self.fut_prices.append((bid_prices[0] + ask_prices[0]) // 2)

        hedge_ratio = self.calculate_hedge_ratio()

        if instrument == Instrument.FUTURE and hedge_ratio is not None and len(self.etf_prices) > 0 and len(self.fut_prices) > 0:
            mid_price = (bid_prices[0] + ask_prices[0]) // 2
            spread = np.array(self.etf_prices) - hedge_ratio * np.array(self.fut_prices)

            if len(spread) > 0 and spread[-1] > np.mean(spread) + np.std(spread):
                new_ask_price = self.adjust_price(mid_price * (1 + CONSERVATIVE_SPREAD))
                new_bid_price = self.adjust_price(mid_price * (1 - NORMAL_SPREAD))
            elif len(spread) > 0 and spread[-1] < np.mean(spread) - np.std(spread):
                new_ask_price = self.adjust_price(mid_price * (1 + NORMAL_SPREAD))
                new_bid_price = self.adjust_price(mid_price * (1 - CONSERVATIVE_SPREAD))
            else:
                new_bid_price = self.adjust_price(mid_price * (1 - NORMAL_SPREAD))
                new_ask_price = self.adjust_price(mid_price * (1 + NORMAL_SPREAD))

            new_ask_price = self.adjust_price(new_ask_price)
            new_bid_price = self.adjust_price(new_bid_price)
            
            exposure_long = len(self.bidq) * LOT_SIZE
            exposure_short = len(self.askq) * LOT_SIZE
            remaining_long = max(0, POSITION_LIMIT - self.position - exposure_long)
            remaining_short = max(0, POSITION_LIMIT + self.position - exposure_short)

            if len(self.bidq) >= 4:
                bid = self.bidq.popleft()
                self.send_cancel_order(bid)
            if len(self.askq) >= 4:
                ask = self.askq.popleft()
                self.send_cancel_order(ask)
            
            if new_bid_price != 0 and remaining_long > 0:
                bid_volume = min(LOT_SIZE, remaining_long)
                bid_id = next(self.order_ids)
                self.send_throttled(bid_id, Side.BUY, new_bid_price, bid_volume)

            if new_ask_price != 0 and remaining_short > 0:
                ask_volume = min(LOT_SIZE, remaining_short)
                ask_id = next(self.order_ids)
                self.send_throttled(ask_id, Side.SELL, new_ask_price, ask_volume)

    def check_hedge(self):
        if self.hedge_time is not None and time.time() - self.hedge_time > HEDGE_TIME_LIMIT and (self.hedge_balance > HEDGE_BOUNDARY or self.hedge_balance < -HEDGE_BOUNDARY):
            if self.hedge_balance > 0:
                self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, abs(self.hedge_balance) - HEDGE_BOUNDARY)
                self.hedge_balance = HEDGE_BOUNDARY
            else:
                self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, abs(self.hedge_balance) - HEDGE_BOUNDARY)
                self.hedge_balance = -HEDGE_BOUNDARY
            self.hedge_time = None
        self.event_loop.call_later(1, self.check_hedge)

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        pre = self.hedge_balance
        if client_order_id in self.bids or client_order_id in self.bidq:
            self.position += volume
            self.hedge_balance -= volume
        elif client_order_id in self.asks or client_order_id in self.askq:
            self.position -= volume
            self.hedge_balance += volume
        if self.hedge_balance >= -HEDGE_BOUNDARY and self.hedge_balance <= HEDGE_BOUNDARY:
            self.hedge_time = None
        elif (pre > 0 and self.hedge_balance < 0) or (pre < 0 and self.hedge_balance > 0):
            self.hedge_time = time.time()     
        elif (self.hedge_balance > HEDGE_BOUNDARY or self.hedge_balance < -HEDGE_BOUNDARY) and self.hedge_time is None:
            self.hedge_time = time.time()

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        if remaining_volume == 0:
            if client_order_id in self.bids or client_order_id in self.bidq:
                self.bids.discard(client_order_id)
                try:
                    self.bidq.remove(client_order_id)
                except ValueError:
                    pass
            elif client_order_id in self.asks or client_order_id in self.askq:
                self.asks.discard(client_order_id)
                try:
                    self.askq.remove(client_order_id)
                except ValueError:
                    pass

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        self.logger.warning("error with order %d: %s",
                            client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        self.logger.info(
            "received trade ticks for instrument %d with sequence number%d", instrument, sequence_number)

    def adjust_price(self, price) -> int:
        adjusted_price = int(price // TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS
        return adjusted_price

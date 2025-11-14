"""Communication module for voice assistant core."""

from .esphome_client import ESPHomeClientWrapper, ESPHomeConnectionError

__all__ = ["ESPHomeClientWrapper", "ESPHomeConnectionError"]

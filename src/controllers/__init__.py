REGISTRY = {}

from .basic_controller import BasicMAC
from .hygma_controller import HYGMA



REGISTRY["basic_mac"] = BasicMAC
REGISTRY["hygma_mac"] = HYGMA

# global_config.py
from typing import Optional

class BrainHeartSettings:
    brain_provider: Optional[str] = None
    brain_model: Optional[str] = None
    heart_provider: Optional[str] = None
    heart_model: Optional[str] = None
    use_premium_search: bool = False
    web_model: Optional[str] = None

settings = BrainHeartSettings()

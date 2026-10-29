"""
Environment defaults for agent config sync (sync_config / async_sync_config).

``SUPERBRYN_API_KEY`` is the key, and the API origin is ``SUPERBRYN_BASE_URL`` (also read by attach),
else ``SUPERBRYN_API_BASE_URL``, else https://api.superbryn.com.
"""

import os

API_CONFIG = {
    "api_key": os.getenv("SUPERBRYN_API_KEY", ""),
    "api_base_url": os.getenv("SUPERBRYN_BASE_URL")
    or os.getenv("SUPERBRYN_API_BASE_URL")
    or "https://api.superbryn.com",
}

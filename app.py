"""ASGI entrypoint: `uvicorn app:app`.

Configuration comes from environment variables / .env (see .env.example).
RETRIEVER_BACKEND=memory (default) runs fully locally from ./docs with no database;
RETRIEVER_BACKEND=atlas uses MongoDB Atlas Vector Search.
"""

import logging

from rag.api import build_service_from_settings, create_app
from rag.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

settings = Settings.from_env()
service, feedback_store, health_check = build_service_from_settings(settings)
app = create_app(service, settings, feedback_store, health_check)

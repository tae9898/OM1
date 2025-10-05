import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import requests

from .singleton import singleton


class MessageType(Enum):
    USER = "user"
    ROBOT = "robot"


@dataclass
class ConversationMessage:
    message_type: MessageType
    content: str
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMessage":
        return cls(
            message_type=MessageType(data.get("type", MessageType.USER.value)),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", 0.0),
        )


@singleton
class TeleopsConversationProvider:

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openmind.org/api/core/teleops/conversation",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.executor = ThreadPoolExecutor(max_workers=1)

    def store_user_message(self, content: str) -> None:
        message = ConversationMessage(
            message_type=MessageType.USER,
            content=content.strip(),
            timestamp=time.time(),
        )
        self._store_message(message)

    def store_robot_message(self, content: str) -> None:
        message = ConversationMessage(
            message_type=MessageType.ROBOT,
            content=content.strip(),
            timestamp=time.time(),
        )
        self._store_message(message)

    def _store_message_worker(self, message: ConversationMessage) -> None:
        if self.api_key is None or self.api_key == "":
            logging.debug("API key is missing. Cannot store conversation message.")
            return

        if not message.content or not message.content.strip():
            logging.debug("Empty content, skipping conversation storage")
            return

        try:
            request = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=message.to_dict(),
                timeout=2,
            )

            if request.status_code == 200:
                logging.debug(
                    f"Successfully stored {message.message_type.value} message to conversation"
                )
            else:
                logging.debug(
                    f"Failed to store {message.message_type.value} message: {request.status_code} - {request.text}"
                )
        except Exception as e:
            logging.debug(
                f"Error storing {message.message_type.value} conversation message: {str(e)}"
            )

    def _store_message(self, message: ConversationMessage) -> None:
        self.executor.submit(self._store_message_worker, message)

    def is_enabled(self) -> bool:
        return self.api_key is not None and self.api_key != ""

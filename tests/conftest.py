# tests/conftest.py
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_openai_chat_model():
    with patch("text2cypher.finetuning.eval.metrics.ChatOpenAI") as mock_chat:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = {"content": "mocked-response"}
        mock_chat.return_value = mock_instance
        yield mock_chat

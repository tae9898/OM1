import logging
import time
import typing as T

import openai
from pydantic import BaseModel

from llm import LLM, LLMConfig
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


class NearAILLM(LLM[R]):
    """
    An NearAI-based Language Learning Model implementation.

    This class implements the LLM interface for Near AI's open-source models, handling
    configuration, authentication, and async API communication.

    Parameters
    ----------
    output_model : Type[R]
        A Pydantic BaseModel subclass defining the expected response structure.
    config : LLMConfig
        Configuration object containing API settings. If not provided, defaults
        will be used.
    """

    def __init__(self, output_model: T.Type[R], config: LLMConfig = LLMConfig()):
        """
        Initialize the NearAI LLM instance.

        Parameters
        ----------
        output_model : Type[R]
            Pydantic model class for response validation.
        config : LLMConfig, optional
            Configuration settings for the LLM.
        """
        super().__init__(output_model, config)

        if not config.api_key:
            raise ValueError("config file missing api_key")
        if not config.model:
            self._config.model = "qwen3-30b-a3b-instruct-2507"

        self._client = openai.AsyncClient(
            base_url=config.base_url or "https://api.openmind.org/api/core/nearai",
            api_key=config.api_key,
        )

        # Initialize history manager
        self.history_manager = LLMHistoryManager(self._config, self._client)

    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, str]] = []
    ) -> R | None:
        """
        Send a prompt to the NearAI API and get a structured response.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        messages : List[Dict[str, str]]
            List of message dictionaries to send to the model.

        Returns
        -------
        R or None
            Parsed response matching the output_model structure, or None if
            parsing fails.
        """
        try:
            logging.info(f"NearAI LLM input: {prompt}")
            logging.info(f"NearAI LLM messages: {messages}")

            self.io_provider.llm_start_time = time.time()

            # Save the input information for debugging
            self.io_provider.set_llm_prompt(prompt)

            response = await self._client.beta.chat.completions.parse(
                model=self._config.model,
                messages=[*messages, {"role": "user", "content": prompt}],
                response_format=self._output_model,
                timeout=self._config.timeout,
            )

            message_content = response.choices[0].message.content
            self.io_provider.llm_end_time = time.time()

            try:
                parsed_response = self._output_model.model_validate_json(
                    message_content
                )
                logging.info(f"NearAI LLM output: {parsed_response}")
                return parsed_response
            except Exception as e:
                logging.error(f"Error parsing NearAI response: {e}")
                return None
        except Exception as e:
            logging.error(f"NearAI API error: {e}")
            return None

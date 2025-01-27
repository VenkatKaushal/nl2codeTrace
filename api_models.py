import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anyscale import Anyscale
from llama_index.llms.cohere import Cohere
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core import Settings
from transformers import AutoTokenizer
import subprocess
from pydantic import BaseModel, Field
from typing import Sequence, Any, List
from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.llms import LLM
from llama_index.core.base.llms.types import TextBlock

from constants import (
    EmbeddingModels,
    EmbeddingModelsMap,
    LLMTypes,
    LLMs,
    LLMsMap,
)



from typing import Sequence, Any, List, Generator, AsyncGenerator
from pydantic import Field
import subprocess

class OllamaLLM(LLM):
    """Ollama Local LLM Integration."""

    model_name: str = Field(default="deepseek-r1:7b", description="The model name for the local LLM")
    max_length: int = Field(default=1024, description="Maximum token length for the LLM")

    def __init__(self, model_name: str = "gemma:2b", max_length: int = 1024):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            model_name=self.model_name,
            max_length=self.max_length,
            supports_streaming=True,
            supports_async=True,
        )

    def convert_chat_messages(self, messages: Sequence[ChatMessage]) -> List[Any]:
        """Convert chat messages to an LLM specific message format."""
        converted_messages = []
        for message in messages:
            if isinstance(message.content, str):
                converted_messages.append(message)
            elif isinstance(message.content, list):
                content_string = ""
                for block in message.content:
                    if isinstance(block, TextBlock):
                        content_string += block.text
                    else:
                        raise ValueError("LLM only supports text inputs")
                message.content = content_string
                converted_messages.append(message)
            else:
                raise ValueError(f"Invalid message content: {message.content!s}")
        return converted_messages

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM."""
        prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
        response_text = self.predict(prompt)
        return ChatResponse(content=response_text)

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint for LLM."""
        response_text = self.predict(prompt)
        return CompletionResponse(text=response_text)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> Generator[ChatResponse, None, None]:
        """Streaming chat endpoint for LLM."""
        prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
        for token in self.stream_predict(prompt):
            yield ChatResponse(content=token)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        """Streaming completion endpoint for LLM."""
        for token in self.stream_predict(prompt):
            yield CompletionResponse(text=token)

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat endpoint for LLM."""
        prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
        response_text = await self.async_predict(prompt)
        return ChatResponse(content=response_text)

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Async completion endpoint for LLM."""
        response_text = await self.async_predict(prompt)
        return CompletionResponse(text=response_text)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        """Async streaming chat endpoint for LLM."""
        prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
        async for token in self.async_stream_predict(prompt):
            yield ChatResponse(content=token)

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> AsyncGenerator[CompletionResponse, None]:
        """Async streaming completion endpoint for LLM."""
        async for token in self.async_stream_predict(prompt):
            yield CompletionResponse(text=token)

    def predict(self, prompt: str) -> str:
        """Run the local LLM with a prompt."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Ollama error: {result.stderr}")
            cleaned_output = result.stdout.replace("<think>", "").replace("</think>", "").strip()
            return cleaned_output
        except Exception as e:
            raise RuntimeError(f"Failed to call Ollama model: {e}")

    def stream_predict(self, prompt: str) -> Generator[str, None, None]:
        """Streaming prediction for LLM."""
        try:
            result = subprocess.Popen(
                ["ollama", "run", "--stream", self.model_name, prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.stdout:
                for line in result.stdout:
                    yield line.strip()
            if result.wait() != 0:
                error = result.stderr.read()
                raise RuntimeError(f"Ollama error: {error}")
        except Exception as e:
            raise RuntimeError(f"Failed to stream call Ollama model: {e}")

    async def async_predict(self, prompt: str) -> str:
        """Asynchronous prediction."""
        return self.predict(prompt)

    async def async_stream_predict(self, prompt: str) -> AsyncGenerator[str, None]:
        """Asynchronous streaming prediction."""
        for token in self.stream_predict(prompt):
            yield token
    async def _as_query_component(self) -> dict:
        """Return a dictionary representation of the LLM for query purposes."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "metadata": self.metadata.model_name,
        }





def get_api_keys(
    api_keys_file: str = 'api_keys.json',
    llm_type: str = LLMTypes.COHERE.value,
    idx=0
):
    """Retrieve API keys for the specified LLM type."""
    all_api_keys = json.load(open(api_keys_file))
    api_key = all_api_keys[llm_type][idx]
    return api_key


def set_embed_model(
    model_name: str = EmbeddingModelsMap[EmbeddingModels.DEFAULT_EMBED_MODEL.value],
):
    """Set the embedding model."""
    embed_llm = HuggingFaceEmbedding(
        model_name=model_name,
        max_length=1024,
    )
    Settings.embed_model = embed_llm


def set_llm(
    model_type: str = None,
    model_name: str = None,
    api_key=None
):
    """Set the LLM based on the specified type and model."""
    api_key = get_api_keys(llm_type=model_type) if api_key is None else api_key

    if model_type == LLMTypes.COHERE.value:
        llm = Cohere(
            model=model_name,
            api_key=api_key,
        )
    elif model_type == LLMTypes.OPENAI.value:
        llm = OpenAI(
            model=model_name,
            api_key=api_key
        )
    elif model_type == LLMTypes.HUGGINGFACE.value:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        llm = HuggingFaceLLM(
            model_name="HuggingFaceH4/zephyr-7b-beta",
            tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
            context_window=3900,
            max_new_tokens=256,
            generate_kwargs={
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            },
            device_map="auto",
        )
    elif model_type == LLMTypes.OLLAMA.value:
        llm = OllamaLLM(
        )
    else:
        raise ValueError(f"Unsupported LLM type: {model_type}")
    
    Settings.llm = llm
    


def set_llm_and_embed(
    llm_type: str = None,
    llm_name: str = None,
    embed_model_name: str = None,
    api_key=None
):
    """Set both the LLM and embedding model."""
    set_llm(llm_type, llm_name, api_key) if llm_type and llm_name else set_llm()
    set_embed_model(embed_model_name) if embed_model_name else set_embed_model()

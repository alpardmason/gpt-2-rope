"""GPT-2 architecture adapted to grouped-query attention and rotary embeddings."""

from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT, CausalLMOutput
from gpt2_rope.tokenizer import ByteBPETokenizer

__all__ = ["GPT", "ByteBPETokenizer", "CausalLMOutput", "ModelConfig"]
__version__ = "0.1.0"

"""Tokenizer for GPT-style model for protein sequences."""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import Fuse
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


class GPTTokenizer(PreTrainedTokenizerFast):

    canonical_aas = list("ACDEFGHIKLMNPQRSTVWY")
    special_aas = list("BXZJUO")

    def __init__(
        self,
        bos_token: str = "<sep>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        eos_token: str | None = None,
        add_special_aas: bool = False,
        additional_special_tokens: list[str] | None = None,
    ):
        # Build Vocab
        vocab = [bos_token] + self.canonical_aas
        if add_special_aas:
            vocab += self.special_aas
        vocab = vocab + [pad_token, unk_token]

        special_tokens = [bos_token, pad_token, unk_token]
        if eos_token:
            vocab.append(eos_token)
            special_tokens.append(eos_token)
        else:
            eos_token = bos_token
        if additional_special_tokens:
            special_tokens += additional_special_tokens
            vocab += additional_special_tokens

        token_to_id = {token: i for i, token in enumerate(vocab)}
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)
        tokenizer = Tokenizer(bpe)

        tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            special_tokens=[
                (bos_token, tokenizer.token_to_id(bos_token)),
                (eos_token, tokenizer.token_to_id(eos_token)),
            ],
        )
        tokenizer.decoder = Fuse()

        super().__init__(
            tokenizer_object=tokenizer,
            bos_token=bos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            eos_token=eos_token,
        )

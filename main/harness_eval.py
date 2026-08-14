from typing import Optional, Union

import os
import torch

from ultra.model import load_model_from_path, load_config_from_path
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
from lm_eval.models.utils import stop_sequences_criteria

@register_model("lingua")
class LinguaLMWrapper(HFLM):
    def __init__(
        self,
        pretrained=None,
        tokenizer=None,
        model_name=None,
        max_length=2048,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        **kwargs,
    ) -> None:
        
        print(f"[LinguaLMWrapper] pretrained: {pretrained}")
        print(f"[LinguaLMWrapper] tokenizer: {tokenizer}")
        print(f"[LinguaLMWrapper] model_name: {model_name}")
        print(f"[LinguaLMWrapper] max_length: {max_length}")
        print(f"[LinguaLMWrapper] device: {device}")
        print(f"[LinguaLMWrapper] dtype: {dtype}")

        self.is_hf = True
        self.model_name = model_name

        super().__init__(
            pretrained=pretrained,
            backend="causal",
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    def _get_config(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:
            self._config = load_config_from_path(self.model_name, pretrained)
            pass

    def _create_model(
        self,
        pretrained: str,
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        **kwargs,
    ) -> None:
            model_name = os.environ.get("MODEL_NAME")
            print(pretrained)
            print(self.model_name)
            print(dtype)
            
            self._model = load_model_from_path(self.model_name, pretrained).to(dtype)
            print(self._model)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
        
if __name__ == "__main__":
    cli_evaluate()

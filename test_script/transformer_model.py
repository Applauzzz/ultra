import torch
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.mistral.configuration_mistral import MistralConfig

import pdb

if __name__ == "__main__":

    config = MistralConfig.from_pretrained("./main/model_config/1B3_baseline/mistral")
    model = MistralForCausalLM(config)
    model = model.to(torch.bfloat16).cuda()
    input_ids = torch.randint(0, 32000, (1, 8 * 2048)).to(torch.long).cuda()
    labels = input_ids
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()

    pdb.set_trace()
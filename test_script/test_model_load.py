import os
from importlib import import_module

MODEL_REGISTRY = {
    "transformer": ("fla.models", "TransformerConfig", "TransformerForCausalLM"),
    "gla": ("fla.models", "GLAConfig", "GLAForCausalLM"),
    "hgrn2": ("fla.models", "HGRN2Config", "HGRN2ForCausalLM"),
    "mamba": ("fla.models", "MambaConfig", "MambaForCausalLM"),
    "mamba2": ("fla.models", "Mamba2Config", "Mamba2ForCausalLM"),
    "gdn": ("fla.models", "GatedDeltaNetConfig", "GatedDeltaNetForCausalLM"),
    "neox": ("transformers", None, "AutoModelForCausalLM"),
}

def load_model_from_config(model_name, config_file):
    module_name, config_cls_name, model_cls_name = MODEL_REGISTRY[model_name]
    module = import_module(module_name)

    if model_name == "neox":
        model_cls = getattr(module, model_cls_name)
        return model_cls.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000"
        )
    else:
        config_cls = getattr(module, config_cls_name)
        model_cls = getattr(module, model_cls_name)
        config = config_cls.from_pretrained(config_file)
        return model_cls(config)

def load_model_from_path(model_name, path):
    module_name, _, model_cls_name = MODEL_REGISTRY[model_name]
    module = import_module(module_name)
    model_cls = getattr(module, model_cls_name)
    return model_cls.from_pretrained(path)

def load_config_from_path(model_name, path):
    module_name, config_cls_name, _ = MODEL_REGISTRY[model_name]
    module = import_module(module_name)
    config_cls = getattr(module, config_cls_name)
    return config_cls.from_pretrained(path)

def get_config(model_name):
    module_name, config_cls_name, _ = MODEL_REGISTRY[model_name]
    module = import_module(module_name)
    config_cls = getattr(module, config_cls_name)
    return config_cls()

def test_all(model_name="gla", config_path="./configs/gla_config.json", model_path="./checkpoints/gla"):
    print(f"\n==== Testing model: {model_name} ====")

    print("load_model_from_config ...")
    model1 = load_model_from_config(model_name, config_path)
    print(f"✅ Loaded model from config: {type(model1)}")
    del model1
    print("get_config ...")
    config_obj = get_config(model_name)
    print(f"✅ Instantiated empty config: {type(config_obj)}")

    print("load_config_from_path ...")
    config_loaded = load_config_from_path(model_name, config_path)
    print(f"✅ Loaded config from path: {type(config_loaded)}")

    print("load_model_from_path ...")
    model2 = load_model_from_path(model_name, model_path)
    print(f"✅ Loaded model from pretrained path: {type(model2)}")


if __name__ == "__main__":
    test_all(model_name="gla",
             config_path="",
             model_path="")

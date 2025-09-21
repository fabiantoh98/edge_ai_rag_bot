import torch
from transformers import BitsAndBytesConfig

from haystack.components.generators import HuggingFaceLocalGenerator

def create_generator(hf_gen_model="HuggingFaceH4/zephyr-7b-beta", bnb_quantize=True):
    if bnb_quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        generator = HuggingFaceLocalGenerator(hf_gen_model,
                                    huggingface_pipeline_kwargs={
                                        "device_map":"auto",
                                        "model_kwargs": {
                                            "quantization_config": bnb_config
                                            }
                                        },
                                        generation_kwargs={"max_new_tokens": 350})
    else:
        generator = HuggingFaceLocalGenerator(hf_gen_model,
                                    huggingface_pipeline_kwargs={
                                        "device_map":"auto"
                                        },
                                        generation_kwargs={"max_new_tokens": 350})
    generator.warm_up()
    return generator
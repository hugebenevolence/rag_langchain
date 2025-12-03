import torch
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_type=torch.float16
)

def get_hf_llm(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens: int = 1024,
    **kwargs
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=nf4_config,  # Disabled temporarily - use quantization_config=nf4_config when bitsandbytes is properly installed
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs=kwargs
    )

    return llm
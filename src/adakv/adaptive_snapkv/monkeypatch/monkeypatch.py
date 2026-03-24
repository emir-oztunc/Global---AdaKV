from importlib.metadata import version
import warnings
import transformers
import transformers.models.Smistral.modeling_mistral

from adaptive_snapkv.monkeypatch.dynamic_llama_hijack import (
    adaptive_LlamaModel_forward as dynamic_LlamaModel_forward,
    adaptive_llama_flash_attn2_forward as dynamic_llama_flash_attn2_forward,
    prepare_inputs_for_generation_llama as dynamic_prepare_inputs,
)

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.37']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")


# config hyperparameters
def config_compress(model, window_size=32, base_capacity=1024, kernel_size=7, pooling="maxpool", floor_alpha=0.5, pyram_mode = False, beta = 20, skip=0, gqa_support=False,gqa_func="mean", out_dir=None):
    model.model.config.window_size = window_size
    model.model.config.base_capacity = base_capacity
    model.model.config.kernel_size = kernel_size

    model.model.config.pooling = pooling
    model.model.config.floor_alpha = floor_alpha
    model.model.config.skip = skip
    model.model.config.normalize = None

    model.model.config.pyram_mode = pyram_mode
    model.model.config.pyram_beta = beta

    model.model.config.gqa_support = gqa_support or (model.config.num_attention_heads != getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads))
    model.model.config.gqa_func = gqa_func
    
    model.model.config.out_dir = out_dir

    return model

def replace_llama_dynamic():
    check_version()
    transformers.models.llama.modeling_llama.LlamaModel.forward = dynamic_LlamaModel_forward
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = dynamic_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = dynamic_prepare_inputs

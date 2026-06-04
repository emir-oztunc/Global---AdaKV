import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
    is_flash_attn_2_available,
)
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from adaptive_snapkv.monkeypatch.dynamic_snapkv_utils import init_adaptive_snapkv, DynamicCacheSplitHeadFlatten, perform_cross_layer_pruning


logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func

def adaptive_LlamaModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # return_legacy_cache = False
    # if (
    #     use_cache and not isinstance(past_key_values, Cache) and not self.training
    # ):  # kept for BC (non `Cache` `past_key_values` inputs)
    #     return_legacy_cache = True
    #     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    #     logger.warning_once(
    #         "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
    #         "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
    #     )

    # NOTE: g-adakv
    return_legacy_cache = False
    if not isinstance(past_key_values, DynamicCacheSplitHeadFlatten):
        past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values)
    logger.warning_once(
        "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
        "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
    )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    hidden_states = hidden_states[:, -1,:].unsqueeze(1)

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def adaptive_llama_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # NOTE: g-adakv
    init_adaptive_snapkv(self)
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

    is_prefill = q_len != 1

    if is_prefill:
        # 1. Calculate the attention scores normally
        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        attn_score = self.kv_cluster.calcul_attn_sore(key_states_rep, query_states)
        
        # 2. DO NOT COMPRESS YET. Store the full, uncompressed cache for now.
        if not hasattr(past_key_value, 'attn_scores'):
            past_key_value.attn_scores = {}
        past_key_value.attn_scores[self.layer_idx] = attn_score # Save this for the pruning step!

        # 3. Store the key/value states fully
        past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ... (keep the rest of the original prefill flash attention code below)

        # repeat k/v heads if n_kv_heads < n_heads
        # [SnapKV] move to ahead
        key_states = key_states_rep
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    else:
        # decoding
        cache_kwargs["head_lens"] = self.kv_cluster.head_lens
        cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen

        if not self.kv_cluster.gqa_support:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # DEBUG: print before cache update (first decode step only)
        _is_first_decode = (self.kv_cluster.max_seqlen_k < 1000) # heuristic for first step
        # if self.layer_idx == 0 and _is_first_decode:
        #     print(f"\n[DEBUG L{self.layer_idx}] === DECODE STEP ===")
        #     print(f"[DEBUG L{self.layer_idx}] key_states (new token) shape: {key_states.shape}, has_nan: {key_states.isnan().any()}, has_inf: {key_states.isinf().any()}")
        #     print(f"[DEBUG L{self.layer_idx}] head_lens: {self.kv_cluster.head_lens}")
        #     print(f"[DEBUG L{self.layer_idx}] cu_klen: {self.kv_cluster.cu_klen}")
        #     print(f"[DEBUG L{self.layer_idx}] klen_sum: {self.kv_cluster.klen_sum}, max_seqlen_k: {self.kv_cluster.max_seqlen_k}")

        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # if self.layer_idx == 0 and _is_first_decode:
        #     print(f"[DEBUG L{self.layer_idx}] After cache update - key_cache shape: {key_states.shape}, has_nan: {key_states.isnan().any()}, has_inf: {key_states.isinf().any()}")
        #     print(f"[DEBUG L{self.layer_idx}] key_cache min: {key_states.min():.4f}, max: {key_states.max():.4f}, mean: {key_states.mean():.4f}")

        # NOTE: update meta data
        self.kv_cluster.klen_sum += self.num_key_value_heads
        self.kv_cluster.max_seqlen_k += 1
        self.kv_cluster.cu_klen = self.kv_cluster.cu_klen.clone() + self.kv_cluster.cu_offset
        self.kv_cluster.head_lens = self.kv_cluster.head_lens.clone() + 1

        # Reshape query to flat format: [num_heads, 1, head_dim] = [32, 1, 128]
        query_states = query_states.view(-1, 1, self.head_dim)
        key_states = key_states.view(-1, 1, self.head_dim)
        value_states = value_states.view(-1, 1, self.head_dim)

        cu_seqlens_k = self.kv_cluster.cu_klen
        num_kv_heads = self.num_key_value_heads
        num_groups = self.num_key_value_groups

        # Manual attention loop: iterate over KV heads.
        # Each KV head serves num_key_value_groups (e.g. 4) query heads.
        attn_outputs = []
        for i in range(num_kv_heads):
            sk, ek = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
            k_i = key_states[sk:ek].transpose(0, 1).transpose(-1, -2)  # [1, head_dim, kv_len]
            v_i = value_states[sk:ek].transpose(0, 1)                   # [1, kv_len, head_dim]

            # Get all query heads that map to this KV head
            q_start = i * num_groups
            q_end = q_start + num_groups
            q_i = query_states[q_start:q_end].transpose(0, 1)  # [1, num_groups, head_dim]

            scores = torch.matmul(q_i, k_i) / math.sqrt(self.head_dim)  # [1, num_groups, kv_len]
            attn_weights = torch.softmax(scores.float(), dim=-1).to(query_states.dtype)

            out_i = torch.matmul(attn_weights, v_i).transpose(0, 1)  # [num_groups, 1, head_dim]
            attn_outputs.append(out_i)

            # DEBUG: print first KV head's attention stats
            # if self.layer_idx == 0 and _is_first_decode and i == 0:
            #     print(f"[DEBUG L{self.layer_idx}] Head 0: kv_len={ek-sk}, scores_range=[{scores.min():.4f}, {scores.max():.4f}], has_nan={scores.isnan().any()}")
            #     print(f"[DEBUG L{self.layer_idx}] Head 0: attn_weights sum={attn_weights.sum(dim=-1)}, out_i range=[{out_i.min():.4f}, {out_i.max():.4f}]")

        attn_output = torch.cat(attn_outputs, dim=0)  # [num_heads, 1, head_dim] = [32, 1, 128]

        # if self.layer_idx == 0 and _is_first_decode:
        #     print(f"[DEBUG L{self.layer_idx}] Final attn_output: shape={attn_output.shape}, has_nan={attn_output.isnan().any()}, has_inf={attn_output.isinf().any()}")
        #     print(f"[DEBUG L{self.layer_idx}] Final attn_output range: [{attn_output.min():.4f}, {attn_output.max():.4f}]")

        #  TODO: support batch size > 1
        assert bsz == 1
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def prepare_inputs_for_generation_llama(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if past_key_values is not None and hasattr(past_key_values, 'attn_scores') and not getattr(past_key_values, 'is_pruned', False):
        
        num_layers = len(past_key_values.attn_scores)
        base_capacity = getattr(self.config, 'base_capacity', 256)
        total_budget = base_capacity * num_layers
        
        # Config options
        floor_ratio = getattr(self.config, 'floor_alpha', 0.2)
        num_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, 'num_key_value_heads', num_heads)
        num_groups = num_heads // num_kv_heads
        
        # For safety/floor
        floor_capacity = int(base_capacity * floor_ratio)
        
        _device = past_key_values.attn_scores[0].device
        bsz = past_key_values.attn_scores[0].shape[0]
        
        dynamic_head_budgets = {}
        
        # 1. Gather all attention scores across layers
        # Each attn_score is [bsz, num_heads, seq_len]
        all_scores_list = []
        for l_idx in range(num_layers):
            attn_score = past_key_values.attn_scores[l_idx]
            # Max/Mean pool across GQA groups to match physical KV heads
            if attn_score.shape[1] == num_heads and num_groups > 1:
                attn_score = attn_score.view(bsz, num_kv_heads, num_groups, -1).max(dim=2).values
            all_scores_list.append(attn_score.unsqueeze(1)) # [bsz, 1, num_kv_heads, seq_len]
            
        # 2. Concatenate globally
        # global_attn has shape [bsz, num_layers, num_kv_heads, seq_len]
        global_attn = torch.cat(all_scores_list, dim=1)
        seq_len = global_attn.shape[-1]
        
        # Flatten to [bsz, num_layers * num_kv_heads * seq_len]
        flat_global_attn = global_attn.reshape(bsz, -1)
        
        # Total global budget for the entire memory pool
        overall_k_val = num_kv_heads * total_budget
        
        # Safety fix: don't prune if budget is bigger than prompt
        if overall_k_val >= flat_global_attn.shape[-1]:
            # Assign max capacity to all
            for l_idx in range(num_layers):
                dynamic_head_budgets[l_idx] = [seq_len for _ in range(num_kv_heads)]
        else:
            # 3. Global Top-K execution
            sorted_global_indices = torch.topk(flat_global_attn, k=overall_k_val, dim=-1).indices
            
            # 4. Decouple token indices back into (layer_idx, head_idx)
            # The flattened index relates to layer*heads*seq_len
            # token_pos = idx % seq_len (we don't strictly need this to count amounts)
            # head_idx = (idx // seq_len) % num_kv_heads
            # layer_idx = (idx // (seq_len * num_kv_heads))
            
            flat_layer_head_idx = sorted_global_indices // seq_len # value between 0 and (num_layers * num_kv_heads - 1)
            
            # Count them precisely
            # bucket_counts shape: [bsz, num_layers * num_kv_heads]
            bucket_counts = torch.zeros((bsz, num_layers * num_kv_heads), device=_device, dtype=sorted_global_indices.dtype)
            bucket_counts.scatter_add_(-1, flat_layer_head_idx, torch.ones_like(flat_layer_head_idx, dtype=bucket_counts.dtype))
            
            # Reshape back to [bsz, num_layers, num_kv_heads]
            bucket_counts = bucket_counts.reshape(bsz, num_layers, num_kv_heads)
            
            # Apply floor constraints globally onto the counts
            bucket_counts = torch.round(bucket_counts * (1 - floor_ratio) + floor_capacity).int()
            
            for l_idx in range(num_layers):
                dynamic_head_budgets[l_idx] = bucket_counts[0, l_idx, :].tolist()
        
        print(f"Global Top-K Budgets Assigned.")
        
        # Execute the actual cache compression using the new global head budgets
        from adaptive_snapkv.monkeypatch.dynamic_snapkv_utils import perform_cross_layer_pruning
        past_key_values = perform_cross_layer_pruning(
            model=self,
            past_key_values=past_key_values,
            dynamic_head_budgets=dynamic_head_budgets,
            config=self.config
        )
        
        past_key_values.is_pruned = True
        
        # Optional: Free up the memory from the stored scores since we don't need them anymore
        del past_key_values.attn_scores

    # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    # Exception 1: when passing input_embeds, input_ids may be missing entries
    # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    if past_key_values is not None:
        if inputs_embeds is not None:  # Exception 1
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

            # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        # The clone here is for the same reason as for `position_ids`.
        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs["input_ids"].shape
            device = model_inputs["input_ids"].device

        dtype = self.lm_head.weight.dtype
        min_dtype = torch.finfo(dtype).min

        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_length(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

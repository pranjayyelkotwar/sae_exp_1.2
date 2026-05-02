# llama_3 module

Concise implementation of Llama 3 text-only inference utilities plus chat/tool formatting and schema definitions. Used by top-level scripts such as capture_activations.

## File map

| File | Core role | Impactful areas |
| --- | --- | --- |
| __init__.py | Package marker (empty). | N/A |
| args.py | Model hyperparameters container (ModelArgs) with validation defaults. | Change model shapes, rope settings, max_seq_len, or vision placeholders. |
| model_text_only.py | PyTorch Llama 3 transformer (attention, FFN, KV cache, rotary embeddings). Adds activation capture and optional SAE hook per layer. | Activation capture (capture_activation, get_layer_residual_activs), SAE integration, KV cache sizing, rotary scaling. |
| tokenizer.py | Tiktoken-based tokenizer with Llama 3 special tokens and encode/decode utilities. | Special token set, max text chunking, tokenizer.model path expectations. |
| chat_format.py | Prompt packing/decoding for chat messages and tool calls; handles images and tool extraction. | Tool parsing/encoding logic, vision token masking, message framing and stop token handling. |
| datatypes.py | Pydantic schemas for roles, messages, tool calls, tool definitions, and media types. | Schema changes affect serialization and tool call parsing compatibility. |
| tool_utils.py | Parse/encode builtin and custom tool calls (json, function_tag, python_list). | Regex patterns, custom tool formats, builtin tool encoding. |
| schema_utils.py | Schema registration helpers and webmethod metadata decorator. | Schema metadata used by datatypes; extend for custom APIs. |

## How to run (integration smoke)

This module has no standalone CLI. The smallest real run path is the activation capture script, which exercises tokenizer, model loading, and forward pass:

```bash
python capture_activations.py \
  --model_name llama_3-8B \
  --dataset_source openwebtext \
  --num_samples 100 \
  --output_dir activation_outs/
```

Notes:
- Requires model artifacts and tokenizer.model in the model directory. The helper in utils downloads supported models.
- Runs distributed; launch with torchrun or your preferred multi-GPU launcher if needed.

## Expected inputs/outputs

- Inputs: tokens from Tokenizer, Message objects for chat formatting, optional images (PIL) for interleaved content.
- Outputs: logits from Transformer forward, tool call structures from ChatFormat decode, optional residual activations per layer.

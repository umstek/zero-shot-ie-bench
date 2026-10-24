# Vendored verbatim from the Hugging Face repo
# notnotsamuel/LFM2.5-350M-RLCD (file rlcd/engine.py), revision
# deb589d803d141cabd158ef55f6617b128529f36. Licenses named on that repo's
# model card: inference code is MIT (LICENSE-CODE); the LiquidAI/LFM2.5-350M
# base weights this engine loads are under the LFM Open License v1.0
# (card license_name: lfm1.0, upstream revision
# 9e6c6ccf47cd318696e137d381a7ded8fe4df09f).
"""Shared prefill + hybrid cache branching + full candidate likelihood scoring."""
import copy
import json
import torch
import jsonschema
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-350M"
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"


def validate_schema(schema):
    jsonschema.Draft202012Validator.check_schema(schema)
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("Only closed, flat object schemas are supported")
    fields = schema.get("properties", {})
    if not fields or set(schema.get("required", [])) != set(fields):
        raise ValueError("All fields must be required")
    if set(schema) - {"type", "properties", "required", "additionalProperties"}:
        raise ValueError("Unsupported object constraints")
    for spec in fields.values():
        if set(spec) - {"type", "enum", "description"}:
            raise ValueError("Unsupported field constraints")
        if spec.get("type") == "boolean" and "enum" not in spec:
            continue
        values = spec.get("enum", [])
        if spec.get("type") != "string" or not values or any(type(v) is not str for v in values):
            raise ValueError("Fields must be booleans or nonempty string enums")
        if len(set(values)) != len(values):
            raise ValueError("Duplicate candidates")


def fork_cache(cache, count):
    """Copy all state and reorder batch rows, including convolution history.

    Generic batch_repeat_interleave is not implemented for LFM2 convolution
    layers in the pinned Transformers release. reorder_cache handles both.
    index_select allocates independent storage; never broadcast mutable views.
    """
    cloned = copy.deepcopy(cache)
    device = next(layer.keys.device for layer in cache.layers if hasattr(layer, "keys"))
    cloned.reorder_cache(torch.zeros(count, dtype=torch.long, device=device))
    return cloned


class Engine:
    def __init__(self, device="mps", dtype="float16"):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=getattr(torch, dtype),
            attn_implementation="eager",
        ).to(device).eval()
        self.model.requires_grad_(False)

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prompt(self, context, schema):
        validate_schema(schema)
        messages = [
            {"role": "system", "content": "Extract the attributes from the text. Return only a JSON object matching this schema. Use the exact allowed values. No explanation or markdown.\n" + json.dumps(schema, ensure_ascii=False)},
            {"role": "user", "content": context},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "{\n"

    def tensor(self, tokens):
        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def constrained(self, context, schema):
        prefix = self.encode(self.prompt(context, schema))
        cache = self.model(self.tensor([prefix]), use_cache=True, logits_to_keep=1).past_key_values
        branches, metadata = [], []
        for name, spec in schema["properties"].items():
            candidates = [True, False] if spec["type"] == "boolean" else spec["enum"]
            # Explicit token boundary before the value; identical in cached and
            # uncached reference evaluation. Full JSON value includes quotes.
            suffix = self.encode("  " + json.dumps(name, ensure_ascii=False) + ": ")
            for candidate in candidates:
                value = self.encode(json.dumps(candidate, ensure_ascii=False) + "\n")
                branches.append(suffix + value)
                metadata.append((name, candidate, len(suffix), value))
        width = max(map(len, branches))
        ids = self.tensor([b + [self.tokenizer.pad_token_id] * (width - len(b)) for b in branches])
        mask = self.tensor([[1] * (len(prefix) + len(b)) + [0] * (width - len(b)) for b in branches])
        out = self.model(ids, past_key_values=fork_cache(cache, len(branches)), attention_mask=mask, use_cache=True)
        scores = []
        # Full likelihood, not first-token proxy. No length normalization.
        for row, (_, _, start, value) in enumerate(metadata):
            logp = out.logits[row, start - 1:start + len(value) - 1].float().log_softmax(-1)
            score = logp.gather(1, self.tensor(value)[:, None]).sum()
            scores.append(score)
        scores = torch.stack(scores).cpu().tolist()
        selected, telemetry = {}, {}
        for name in schema["properties"]:
            options = [(m[1], s) for m, s in zip(metadata, scores) if m[0] == name]
            selected[name] = max(options, key=lambda x: x[1])[0]
            telemetry[name] = [{"value": v, "log_likelihood": s} for v, s in options]
        return {"text": json.dumps(selected, ensure_ascii=False, allow_nan=False),
                "scores": telemetry, "prompt_tokens": len(prefix), "branches": len(branches),
                "branch_tokens_padded": len(branches) * width, "forward_calls": 2}

    @torch.inference_mode()
    def autoregressive(self, context, schema, max_new_tokens=192):
        prefix = self.encode(self.prompt(context, schema))
        ids = self.tensor([prefix])
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids),
                                  do_sample=False, max_new_tokens=max_new_tokens,
                                  pad_token_id=self.tokenizer.pad_token_id,
                                  eos_token_id=self.tokenizer.eos_token_id)
        continuation = out[0, len(prefix):].tolist()
        return {"text": "{\n" + self.tokenizer.decode(continuation, skip_special_tokens=True),
                "prompt_tokens": len(prefix), "generated_tokens": len(continuation),
                "hit_token_limit": len(continuation) == max_new_tokens and continuation[-1] != self.tokenizer.eos_token_id}

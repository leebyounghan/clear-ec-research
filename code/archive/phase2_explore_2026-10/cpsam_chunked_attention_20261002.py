"""Exact global SAM attention with query-only chunking.

This changes execution/memory scheduling, not the attention neighborhood. All
keys/values and both query-dependent decomposed relative-position terms remain.
No weights, parameter names, or positional tables are replaced. Runtime parity
must be checked in the actual installed SAM version before training/deployment.
"""
from copy import deepcopy
import json
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint


def _chunk(q, k, v, rh, rw, scale):
    # SAM scales Q only for QK, NOT for the decomposed relative-position terms.
    logits = (q * scale) @ k.transpose(-2, -1)
    if rh is not None:
        rel_h = torch.einsum("bqd,qkd->bqk", q, rh)
        rel_w = torch.einsum("bqd,qkd->bqk", q, rw)
        logits = (logits.reshape(q.shape[0], q.shape[1], rh.shape[1], rw.shape[1])
                  + rel_h[:, :, :, None] + rel_w[:, :, None, :])
        logits = logits.flatten(-2)
    return logits.softmax(dim=-1) @ v


def chunked_forward(self, x):
    from segment_anything.modeling.image_encoder import get_rel_pos

    batch, height, width, channels = x.shape
    tokens = height * width
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, batch * self.num_heads, tokens, -1).unbind(0)
    rh = rw = None
    if self.use_rel_pos:
        # Preserve the original SAM interpolation and distance indexing.
        rh = get_rel_pos(height, height, self.rel_pos_h)
        rw = get_rel_pos(width, width, self.rel_pos_w)
    chunks = []
    for start in range(0, tokens, self._clear_ec_query_chunk):
        stop = min(start + self._clear_ec_query_chunk, tokens)
        qc = q[:, start:stop]
        rhc = rwc = None
        if rh is not None:
            global_query = torch.arange(start, stop, device=x.device)
            rhc = rh[torch.div(global_query, width, rounding_mode="floor")]
            rwc = rw[global_query.remainder(width)]
        if self._clear_ec_checkpoint_chunks and self.training and torch.is_grad_enabled():
            # Passing every differentiable operand explicitly also avoids a
            # late-binding loop closure when backward recomputes earlier chunks.
            out = checkpoint(_chunk, qc, k, v, rhc, rwc, self.scale,
                             use_reentrant=False, preserve_rng_state=True)
        else:
            out = _chunk(qc, k, v, rhc, rwc, self.scale)
        chunks.append(out)
    x = torch.cat(chunks, dim=1).reshape(batch, self.num_heads, height, width, -1)
    x = x.permute(0, 2, 3, 1, 4).reshape(batch, height, width, channels)
    return self.proj(x)


def install_attention(attention, query_chunk=256, checkpoint_chunks=False):
    """Bind a forward method without changing any state_dict key or parameter.

    Intended for the upstream segment_anything Attention class only. Do not
    apply to DINO attention, flash-SAM forks, or modules with additional masks.
    Use eval()+inference_mode() for memory-bounded inference. In training, enable
    chunk checkpointing; otherwise autograd retains all chunks' attention maps.
    """
    from segment_anything.modeling.image_encoder import Attention

    if type(attention) is not Attention:
        raise TypeError(f"Expected upstream SAM Attention, got {type(attention)}")
    if isinstance(query_chunk, bool) or not isinstance(query_chunk, int) or query_chunk <= 0:
        raise ValueError("query_chunk must be a positive integer")
    if not hasattr(attention, "_clear_ec_original_forward"):
        attention._clear_ec_original_forward = attention.forward
    attention._clear_ec_query_chunk = query_chunk
    attention._clear_ec_checkpoint_chunks = bool(checkpoint_chunks)
    attention.forward = MethodType(chunked_forward, attention)
    return attention


def install_encoder(encoder, query_chunk=256, checkpoint_chunks=False):
    """Patch every SAM block's attention; preserve original window partitioning."""
    for block in encoder.blocks:
        install_attention(block.attn, query_chunk, checkpoint_chunks)
    return encoder


def restore_attention(attention):
    if hasattr(attention, "_clear_ec_original_forward"):
        attention.forward = attention._clear_ec_original_forward
        del attention._clear_ec_original_forward
        del attention._clear_ec_query_chunk
        del attention._clear_ec_checkpoint_chunks


def parity_check(device="cpu", atol=3e-5, rtol=3e-4):
    """FP32 output/input/ALL parameter-gradient tests against upstream blocks.

    Includes nonzero relative tables, interpolation, rectangular grids, windows,
    tail chunks, and checkpoint backward. No trained or published model needed.
    This unit test is not whole-encoder or low-precision deployment validation.
    """
    from segment_anything.modeling.image_encoder import Attention, Block

    torch.manual_seed(10202)
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for use_block, use_rel, window, height, width, chunk in (
        (False, True, 0, 7, 11, 13),
        (False, False, 0, 8, 8, 17),
        (True, True, 0, 7, 11, 13),
        (True, True, 4, 7, 11, 7),
    ):
        if use_block:
            reference = Block(dim=32, num_heads=4, mlp_ratio=2,
                              use_rel_pos=use_rel, window_size=window,
                              input_size=(5, 6)).to(device)
            target_attention = reference.attn
        else:
            reference = Attention(dim=32, num_heads=4, use_rel_pos=use_rel,
                                  input_size=(5, 6)).to(device)
            target_attention = reference
        if use_rel:
            with torch.no_grad():
                target_attention.rel_pos_h.normal_(std=.1)
                target_attention.rel_pos_w.normal_(std=.1)
        candidate = deepcopy(reference)
        install_attention(candidate.attn if use_block else candidate,
                          query_chunk=chunk, checkpoint_chunks=True)
        assert list(reference.state_dict()) == list(candidate.state_dict())
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
        inputs = torch.randn(2, height, width, 32, device=device)
        original_input = inputs.detach().clone().requires_grad_(True)
        candidate_input = inputs.detach().clone().requires_grad_(True)
        reference.train()
        candidate.train()
        original_output, candidate_output = reference(original_input), candidate(candidate_input)
        torch.testing.assert_close(original_output, candidate_output, atol=atol, rtol=rtol)
        target = torch.randn_like(original_output)
        (original_output * target).sum().backward()
        (candidate_output * target).sum().backward()
        torch.testing.assert_close(original_input.grad, candidate_input.grad, atol=atol, rtol=rtol)
        candidate_parameters = dict(candidate.named_parameters())
        gradient_error = 0.
        for name, parameter in reference.named_parameters():
            candidate_grad = candidate_parameters[name].grad
            assert parameter.grad is not None and candidate_grad is not None, name
            torch.testing.assert_close(parameter.grad, candidate_grad, atol=atol, rtol=rtol,
                                       msg=lambda msg: f"{name}: {msg}")
            gradient_error = max(gradient_error, float((parameter.grad-candidate_grad).abs().max()))
        candidate.eval()
        reference.eval()
        with torch.inference_mode():
            torch.testing.assert_close(reference(inputs), candidate(inputs), atol=atol, rtol=rtol)
        results.append(dict(block=use_block, relative=use_rel, window=window,
                            grid=[height, width], query_chunk=chunk,
                            output_max_abs=float((original_output-candidate_output).abs().max()),
                            input_gradient_max_abs=float((original_input.grad-candidate_input.grad).abs().max()),
                            parameter_gradient_max_abs=gradient_error, status="passed"))
    return dict(device=str(device), dtype="float32", tests=results,
                caveat="Small synthetic unit tests only; whole-encoder and FP16/BF16 checks required")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(parity_check(args.device), indent=2))

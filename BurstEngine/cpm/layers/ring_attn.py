import torch
import bmtrain as bmt
from einops import rearrange
from .comm import all_reduce, ring_bmt

def _calc_incoming_device_range(i, rank, world_size, sub_seq_length):
    device_of_incoming_k = (rank - i - 1) % world_size
    start_idx = sub_seq_length * device_of_incoming_k
    end_idx = sub_seq_length * (device_of_incoming_k + 1)
    return start_idx, end_idx
def _calc_current_device_range(rank, sub_seq_length):
    start_idx = sub_seq_length * rank
    end_idx = sub_seq_length * (rank + 1)
    return start_idx, end_idx

class RingQK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sub_q, sub_k, batch_size, num_attention_heads, sub_seq_length, softmax_scale, mask_bias):
        ctx.save_for_backward(sub_q, sub_k)
        ctx.sub_seq_length = sub_seq_length

        attention_score = torch.empty(batch_size * num_attention_heads,
                                      sub_seq_length,
                                      sub_seq_length * bmt.world_size(),
                                      dtype=sub_q.dtype,
                                      device="cuda")
        ctx.softmax_scale = softmax_scale
        ctx.mask_bias = mask_bias

        i = bmt.rank()
        part_a = torch.matmul(sub_q, sub_k.transpose(2, 1)) * softmax_scale
        if ctx.mask_bias is not None:
            part_a = (part_a.view(ctx.mask_bias.shape[0], -1, *part_a.shape[-2:]) + ctx.mask_bias[:,:,i*ctx.sub_seq_length:(i+1)*ctx.sub_seq_length, i*ctx.sub_seq_length:(i+1)*ctx.sub_seq_length]).view(part_a.shape)
        attention_score[:, :, i*ctx.sub_seq_length:(i+1)*ctx.sub_seq_length] = part_a

        i = bmt.rank()
        for k in range(1, bmt.world_size()):
            j = (i + bmt.world_size() - k) % bmt.world_size()
            sub_k = ring_bmt(sub_k)
            part_a = torch.matmul(sub_q, sub_k.transpose(2, 1)) * softmax_scale
            if ctx.mask_bias is not None:
                part_a = (part_a.view(ctx.mask_bias.shape[0], -1, *part_a.shape[-2:]) + ctx.mask_bias[:,:,i*ctx.sub_seq_length:(i+1)*ctx.sub_seq_length, j*ctx.sub_seq_length:(j+1)*ctx.sub_seq_length]).view(part_a.shape)
            attention_score[:, :, j*ctx.sub_seq_length:(j+1)*ctx.sub_seq_length] = part_a
        return attention_score

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.softmax_scale
        sub_q, sub_k, = ctx.saved_tensors
        i = bmt.rank()

        grad_k = torch.matmul(grad_output.transpose(2, 1), sub_q)

        all_reduce(grad_k)
        grad_k = grad_k[:, i*ctx.sub_seq_length:(i+1)*ctx.sub_seq_length]
        grad_k /= bmt.world_size()

        grad_q = torch.zeros_like(
            sub_q,
            dtype=sub_q.dtype,
            device="cuda"
        )

        grad_q += torch.matmul(grad_output[:, :, i*ctx.sub_seq_length:(i+1)*ctx.sub_seq_length], sub_k)

        for k in range(1, bmt.world_size()):
            j = (i + bmt.world_size() - k) % bmt.world_size()
            sub_k = ring_bmt(sub_k)
            grad_q += torch.matmul(grad_output[:, :, j*ctx.sub_seq_length:(j+1)*ctx.sub_seq_length], sub_k)

        grad_q /= bmt.world_size()

        return grad_q, grad_k, None, None, None, None, None


class RingAV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, attention_score, sub_v, batch_size, num_attention_heads, attention_head_size, sub_seq_length):
        local_rank = bmt.rank()
        local_world_size = bmt.world_size()
        local_start_idx, local_end_idx = _calc_current_device_range(local_rank, sub_seq_length)

        sub_attention_result = torch.zeros(batch_size * num_attention_heads,
                                           sub_seq_length,
                                           attention_head_size,
                                           device="cuda",
                                           dtype=attention_score.dtype)

        ctx.save_for_backward(attention_score, sub_v)
        ctx.sub_seq_lengthgth = sub_seq_length

        part_av = torch.matmul(attention_score[:, :, local_start_idx:local_end_idx], sub_v)
        sub_attention_result += part_av

        for i in range(local_world_size - 1):
            sub_v = ring_bmt(sub_v)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, sub_seq_length)

            # compute QK^T
            part_av = torch.matmul(attention_score[:, :, start_idx:end_idx], sub_v)
            sub_attention_result += part_av
        return sub_attention_result

    @staticmethod
    def backward(ctx, grad_output):
        local_rank = bmt.rank()
        local_world_size = bmt.world_size()
        local_start_idx, local_end_idx = _calc_current_device_range(local_rank, ctx.sub_seq_lengthgth)
        attention_scores, sub_v = ctx.saved_tensors

        # calculate gradient of v
        grad_v = torch.matmul(attention_scores.transpose(2, 1), grad_output)
        all_reduce(grad_v)
        grad_v = grad_v[:, local_start_idx:local_end_idx]
        grad_v /= local_world_size

        # calculate gradient for attention score
        grad_attention_score = torch.zeros_like(attention_scores, dtype=grad_output.dtype, device="cuda")

        # compute with local sub_k
        grad_attention_score[:, :, local_start_idx:local_end_idx] += torch.matmul(grad_output, sub_v.transpose(2, 1))

        # compute QK^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_v = ring_bmt(sub_v)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, ctx.sub_seq_lengthgth)

            # compute grad_q
            grad_attention_score[:, :, start_idx:end_idx] += torch.matmul(grad_output, sub_v.transpose(2, 1))
        return grad_attention_score, grad_v, None, None, None, None
def ring_attn(q,k,v, sm_scale=1.0, mask_bias=None):
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    sub_seq = q.shape[2]
    hidden_dim = q.shape[-1]
    q = q.flatten(0,1)
    k = k.flatten(0,1)
    v = v.flatten(0,1)
    attn_score = RingQK.apply(q,k,batch_size,num_heads,sub_seq,sm_scale,mask_bias)
    attn_score = torch.softmax(attn_score, dim=-1)
    if mask_bias is not None:
        attn_score = torch.masked_fill(
            attn_score.view(mask_bias.shape[0], -1, *attn_score.shape[-2:]),
            mask_bias[:, :, bmt.rank()*sub_seq:(bmt.rank()+1)*sub_seq] == -10000,
            torch.scalar_tensor(0, device=attn_score.device, dtype=attn_score.dtype),
        ).view(attn_score.shape)
    out = RingAV.apply(attn_score,v,batch_size,num_heads,hidden_dim,sub_seq)
    out = rearrange(out, "(b n) s d -> b n s d", b=batch_size)
    return out
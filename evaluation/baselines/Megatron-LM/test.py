import torch
import bmtrain as bmt
loss_func1 = torch.nn.CrossEntropyLoss()
loss_func2 = bmt.loss.FusedCrossEntropy()
batch = 4
seqlen = 128
vocab = 1024
inp = torch.randn(batch, seqlen, vocab, dtype=torch.float32, device="cuda")
tgt = torch.randint(0, vocab, (batch, seqlen), dtype=torch.long, device="cuda")
tgt[0][1:] = -100
tgt[1][2:] = -100
tgt[2][3:] = -100
tgt[3][4:] = -100
loss1 = loss_func1(inp.transpose(1, 2), tgt)
inp2 = inp.to(dtype=torch.float16)
loss2 = loss_func2(inp2.reshape(batch * seqlen, vocab), tgt.reshape(batch * seqlen))
from IPython import embed;embed()

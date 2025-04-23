## burst
g43: burst
g43:       batch_size   |     seqlen     |   num_heads    |      dim       |     causal     |  double_ring   |    opt_bwd   
  
g43:           1        |     131072     |       40       |      128       |      True      |      True      |      True    
  
g43:    Time: | forward: 0.05 s | forward_backward: 0.15 s
g43:    TFLOPS| | forward: 112.30 TFLOPS | forward_backward: 129.44 TFLOPS | backward: 137.71 TFLOPS
g43: burst
g43:       batch_size   |     seqlen     |   num_heads    |      dim       |     causal     |  double_ring   |    opt_bwd   
  
g43:           1        |     262144     |       40       |      128       |      True      |      True      |      True    
  
g43:    Time: | forward: 0.13 s | forward_backward: 0.47 s
g43:    TFLOPS| | forward: 170.25 TFLOPS | forward_backward: 165.48 TFLOPS | backward: 163.66 TFLOPS
g43: burst
g43:       batch_size   |     seqlen     |   num_heads    |      dim       |     causal     |  double_ring   |    opt_bwd   
  
g43:           1        |     524288     |       40       |      128       |      True      |      True      |      True    
  
g43:    Time: | forward: 0.47 s | forward_backward: 1.72 s
g43:    TFLOPS| | forward: 187.65 TFLOPS | forward_backward: 179.13 TFLOPS | backward: 175.69 TFLOPS


## Double_Ring
g43: =======================================================================================================================
=
g43:  Dtype   | QKV Format |  B   | Seq Len |  CP   | Window  |  FWD (ms)  | FWD+BWD (ms) | Memory (MB)  |  TFLOPs/s 
g43: -----------------------------------------------------------------------------------------------------------------------
-
g43:   bf16   |   bshd   |  1   | 131072  | True  |  None   |   46.62    |    148.48    |   1241.25    |   129.59  
g43:   bf16   |   bshd   |  1   | 262144  | True  |  None   |   137.45   |    467.88    |   2482.50    |   164.50  
g43:   bf16   |   bshd   |  1   | 524288  | True  |  None   |   471.33   |   1939.78    |   4965.00    |   158.71  
g43: =======================================================================================================================
=

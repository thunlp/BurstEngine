g43:                                                                                                 Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
g43: ----------------------------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
g43: void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, ...         0.00%       0.000us         0.00%       0.000us       0.000us       14.531s        44.46%       14.531s      37.452ms           388  
g43:                                             ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)         0.00%       0.000us         0.00%       0.000us       0.000us       11.306s        34.59%       11.306s       9.336ms          1211  
g43: void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t,...         0.00%       0.000us         0.00%       0.000us       0.000us        5.398s        16.52%        5.398s      14.511ms           372  
g43: void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, ...         0.00%       0.000us         0.00%       0.000us       0.000us     451.602ms         1.38%     451.602ms      37.633ms            12  
g43: void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, at:...         0.00%       0.000us         0.00%       0.000us       0.000us     230.334ms         0.70%     230.334ms     241.440us           954  
g43: void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t,...         0.00%       0.000us         0.00%       0.000us       0.000us     166.490ms         0.51%     166.490ms      13.874ms            12  
g43: void flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, fal...         0.00%       0.000us         0.00%       0.000us       0.000us     142.506ms         0.44%     142.506ms     356.265us           400  
g43:                                                                     fused_to_sub_sigmoid_sub_mul_sub         0.00%       0.000us         0.00%       0.000us       0.000us     128.383ms         0.39%     128.383ms     345.117us           372  
g43: void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<float>, at::detail::Arra...         0.00%       0.000us         0.00%       0.000us       0.000us     120.450ms         0.37%     120.450ms     477.977us           252  
g43:                                                                       Memcpy DtoD (Device -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us      85.837ms         0.26%      85.837ms     230.744us           372  
g43: void flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, ...         0.00%       0.000us         0.00%       0.000us       0.000us      65.489ms         0.20%      65.489ms     163.314us           401  
g43: void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterat...         0.00%       0.000us         0.00%       0.000us       0.000us      21.758ms         0.07%      21.758ms     453.298us            48  
g43: void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterat...         0.00%       0.000us         0.00%       0.000us       0.000us      20.566ms         0.06%      20.566ms     428.459us            48  
g43: void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct...         0.00%       0.000us         0.00%       0.000us       0.000us       7.407ms         0.02%       7.407ms       9.746us           760  
g43: void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFu...         0.00%       0.000us         0.00%       0.000us       0.000us       6.445ms         0.02%       6.445ms       8.662us           744  
g43: void at::native::vectorized_elementwise_kernel<4, at::native::launch_log_sigmoid_forward_kernel(a...         0.00%       0.000us         0.00%       0.000us       0.000us       2.476ms         0.01%       2.476ms       6.656us           372  
g43:                                                                                       cudaEventQuery         0.41%      91.541ms         0.42%      92.752ms       1.504us       0.000us         0.00%       0.000us       0.000us         61678  
g43:                                                                                      cudaEventRecord         0.05%      10.318ms         0.05%      10.404ms       1.227us       0.000us         0.00%       0.000us       0.000us          8480  
g43:                                                                                  cudaStreamWaitEvent         0.03%       7.042ms         0.03%       7.057ms       1.029us       0.000us         0.00%       0.000us       0.000us          6860  
g43:                                                                          cudaStreamGetCaptureInfo_v2         0.00%     906.942us         0.00%     909.577us       0.928us       0.000us         0.00%       0.000us       0.000us           980  
g43:                                                                                cudaStreamIsCapturing         0.01%       2.838ms         0.01%       2.842ms       0.580us       0.000us         0.00%       0.000us       0.000us          4900  
g43:                                                                                  cudaGetFuncBySymbol         0.00%     462.982us         0.00%     464.167us       0.474us       0.000us         0.00%       0.000us       0.000us           980  
g43:                                                                                     cuLaunchKernelEx         0.03%       7.647ms         0.04%       8.018ms       8.182us       0.000us         0.00%       0.000us       0.000us           980  
g43:                                                                                 cudaFuncSetAttribute         0.01%       1.276ms         0.01%       1.278ms       1.996us       0.000us         0.00%       0.000us       0.000us           640  
g43:                                                                                     cudaLaunchKernel        70.50%       15.572s        70.63%       15.600s       4.031ms       0.000us         0.00%       0.000us       0.000us          3870  
g43:                                                                                       cuLaunchKernel         0.07%      15.575ms         0.07%      15.970ms      51.516us       0.000us         0.00%       0.000us       0.000us           310  
g43:                                                                                      cudaMemcpyAsync         9.41%        2.078s         9.45%        2.087s       6.732ms       0.000us         0.00%       0.000us       0.000us           310  
g43:                                                                               cudaDeviceGetAttribute         0.00%     272.779us         0.00%     272.779us       0.852us       0.000us         0.00%       0.000us       0.000us           320  
g43:                                                                                cudaDeviceSynchronize        19.47%        4.301s        19.51%        4.310s        4.310s       0.000us         0.00%       0.000us       0.000us             1  
g43: ----------------------------------------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
g43: Self CPU time total: 22.088s
g43: Self CUDA time total: 32.686s
g43: 
g43: {'flash': [(451.602, 0.0138), (166.49, 0.0051), (142.506, 0.0044), (65.489, 0.002)], 'SendRecv': []}
g43: inter intra
g43: [0, 8, 16, 24]     [0, 1, 2, 3, 4, 5, 6, 7]
g43: ==================================================
g43: Benchmark Results for ZigZagRingFlashAttnFunc
g43: ==================================================
g43: Configuration:
g43:   - Dtype: bf16
g43:   - QKV Format: bshd
g43:   - Batch Size: 1
g43:   - Sequence Length: 524288
g43:   - Num Heads: 40
g43:   - Head Dimension: 128
g43:   - GQA Groups: 40
g43:   - Attention Mask: causal
g43:   - Window Size: (-1, -1)
g43:   - Context Parallel: True
g43:   - World Size: 32
g43: 
g43: Performance:
g43:   - Forward Time: 530.17 ms
g43:   - Forward+Backward Time: 1941.49 ms
g43:   - FlashAttention Kernel: 0.03 %
g43:   - RingComm : 0.00 %
g43:   - Memory Usage: 4485.00 MB
g43:   - Performance: 158.57 TFLOPs/s
g43: ==================================================
g43: 
g43: 
g43: Benchmark Summary:
g43: ========================================================================================================================
g43:  Dtype   | QKV Format |  B   | Seq Len |  CP   | Window  |  FWD (ms)  | FWD+BWD (ms) | Memory (MB)  |  TFLOPs/s 
g43: ------------------------------------------------------------------------------------------------------------------------
g43:   bf16   |   bshd   |  1   | 524288  | True  |  None   |   530.17   |   1941.49    |   4485.00    |   158.57  
g43: ========================================================================================================================

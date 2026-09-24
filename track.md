- [ ] python==3.11.1
- [x] 配置解耦
- [x] 调用壳
-- [x] 输出流
-- [x] 备份配置
-- [x] 转移输出
- [x] git和track
-- [x] 建立git
-- [x] 同步cloud studio
-- [x] 重新梳理总体的优化步骤
- [x] 复现优化
-- [x] baseline通过
--- [x] 依赖列表恢复
--- [-] 通过依赖列表测试环境复现
---- [-] 尝试找出torchmetrics修改的库并进行限制，而不是全部requirements
--- [x] 跳过baseline环境的requirements清单构建（本身那个环境pip check都通过不了）
--- [x] 将输入变形器适配旧版torchvision
-- [x] 去掉items
-- [x] 测试batch size
-- [x] 测试num workers
-- [x] 测试pin mem
- [x] 节点解读profiler并添加待办
-- [x] 解读
-- [x] loss.detach并校准batch size累计
-- [x] 调整batch size使全部batch相等并去除校准
-- [x] 分离首个epoch和后续epoch 的profiler
-- [x] 自动cuDNN卷积测量
-- [x] 减少优化器中零散的内核启动
-- [x] 输出uint8图片，传到GPU上进行normalize
--- [x] 取消epoch之间的同步
--- [x] 撤销↑
-- [x] 全部搬到cuda cudaif（缓存离线预处理）
-- [x] 增加nhwc（channel last）
-- [x] 关闭抗锯齿
-- [x] 升级transforms管线到v2标准
-- [x] 和之前的屎山配置进行对比，找出为什么不如之前的原因
--- [x] 取消循环中cuda_if的分支，然而并没有用（循环里的if真的没有那么慢呢）
--- [x] 取消Adam的fused参数，成功
-- [x] 取消gpu transform
-- [x] 增加AMP
-- [x] accuracy避免每个batch都compute
- [x] 测试100%数据集

1. 安装命令：
/root/.pyenv/versions/3.11.1/bin/python -m pip install --no-deps "numpy>=1.26.4,<2.3" "scipy>=1.13,<1.17" "torchmetrics==1.9.0" "lightning-utilities==0.15.3" "torch==2.6.0" "torchvision==0.21.0" "transformers==5.17.0"
/root/.pyenv/versions/3.11.1/bin/python -m pip install omegaconf

2.  实验记录，统一8%*4epoch
baseline：36.2s
优化1. 将items从循环中提出，28.4s
优化2. num workers=7，24.4s（其实对于小数据集，num workers应搭配persistent workers减小重建开销）
经过MNIST测试，num workers分别为2 4 6 7 8时，对应时间大约是12 8 6.8 6.5 6.7s，可能是主进程、OpenCV、NumPy以及其他系统进程需要，6或7个num workers效果最好
优化3. pin memory=true, 然后搭配non_blocking=true, 再加上上一步忘记的persistent workers=true，22.8s
优化4. detach loss，并增加batch size求和校准loss计算，23.8s
优化5. 调整batch size使全部batch相等以此去除校准loss计算，21.3s
分离首个epoch和后续epoch 的profiler后，时间变为20.5s（暂不清楚原因）
[deprecated]自动cuDNN卷积测量
优化6. 减少优化器中零散的内核启动，20.2s
优化7. 输出uint8图片，传到GPU上进行normalize，19.4s
优化8. 改为NHWC，16.7s
优化9. 关闭antialias，16.3s
优化10. 升级transforms管线到v2标准，16.1s
优化11. 取消Adam的fused参数，使用默认，18.8s
以后测速一定要关掉profiler，开着的时候测时间是不准的
优化12. autocast，12.6s
打开test开关，20%测试集，12.9s
以上都是在开profiler的情况下测的，最后关掉profiler，5.29s




3. 一些讨论

将loss的item提出到循环外前：
enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        46.92%       12.459s        56.52%       15.007s     937.958ms       0.000us         0.00%       0.000us       0.000us            16  
                                  cudaStreamSynchronize        33.70%        8.948s        33.70%        8.948s     248.565ms       0.000us         0.00%       0.000us       0.000us            36  
                       Runtime Triggered Module Loading         4.81%        1.278s         4.81%        1.278s      17.747ms     131.883ms         1.23%     131.883ms       1.832ms            72  
                                             aten::mul_         2.50%     662.861ms         2.56%     678.505ms     588.980us       0.000us         0.00%       0.000us       0.000us          1152  
                                              aten::sub         2.17%     575.693ms         2.17%     575.693ms     499.733us       0.000us         0.00%       0.000us       0.000us          1152  
                                       cudaLaunchKernel         2.10%     558.034ms         4.94%        1.310s     146.206us       0.000us         0.00%     197.221ms      22.006us          8962  
                                            aten::copy_         2.10%     556.871ms         2.74%     727.218ms     312.379us     165.505ms         1.54%     165.505ms      71.093us          2328  
                                             aten::div_         1.70%     451.890ms         1.70%     451.890ms     392.265us       0.000us         0.00%       0.000us       0.000us          1152  
                                              aten::cat         0.82%     216.410ms         0.82%     216.632ms      18.053ms       0.000us         0.00%       0.000us       0.000us            12  

后：
enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        55.87%       12.570s        70.79%       15.925s     995.311ms       0.000us         0.00%       0.000us       0.000us            16  
                                        cudaMemcpyAsync        14.09%        3.171s        14.09%        3.171s     132.104ms       0.000us         0.00%       0.000us       0.000us            24  
                       Runtime Triggered Module Loading         5.71%        1.284s         5.71%        1.284s      17.828ms     131.900ms         1.23%     131.900ms       1.832ms            72  
                                  cudaDeviceSynchronize         4.29%     966.103ms         4.29%     966.103ms     241.526ms       0.000us         0.00%       0.000us       0.000us             4  
                                             aten::mul_         3.43%     771.864ms         3.50%     787.526ms     683.617us       0.000us         0.00%       0.000us       0.000us          1152  
                                              aten::sub         3.17%     712.445ms         3.17%     712.445ms     618.442us       0.000us         0.00%       0.000us       0.000us          1152  
                                            aten::copy_         3.11%     700.551ms        17.21%        3.871s       1.663ms     160.669ms         1.49%     160.669ms      69.016us          2328  
                                             aten::div_         2.57%     577.426ms         2.57%     577.426ms     501.238us       0.000us         0.00%       0.000us       0.000us          1152  
                                       cudaLaunchKernel         2.49%     560.147ms         5.84%        1.315s     146.496us       0.000us         0.00%     197.285ms      21.984us          8974  
                                              aten::cat         2.23%     501.991ms         2.23%     502.218ms      41.852ms       0.000us         0.00%       0.000us       0.000us            12  

分析：显著降低了StreamSynchronize的时间，说明CPU等待的问题得到解决，但是内存复制成为新的瓶颈（推测因为StreamSynchronize的时候，内存复制同时发生，所以被前者包含了，修改前看不到内存复制的时间）。下一步先考虑降低enumerate(DataLoader)的时间，也就是说CPU加载数据的速度太慢了。

给程序加上了numworkers=7，profiler如下：
                                        cudaMemcpyAsync        43.66%        8.171s        43.66%        8.171s     340.446ms       0.000us         0.00%       0.000us       0.000us            24  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        39.25%        7.346s        39.25%        7.346s     459.153ms       0.000us         0.00%       0.000us       0.000us            16  
                       Runtime Triggered Module Loading         6.66%        1.247s         6.66%        1.247s      17.319ms     132.017ms         1.23%     132.017ms       1.834ms            72  
                                  cudaDeviceSynchronize         4.18%     782.866ms         4.18%     782.866ms     195.716ms       0.000us         0.00%       0.000us       0.000us             4  
                                       cudaLaunchKernel         2.88%     539.356ms         6.86%        1.284s     143.069us       0.000us         0.00%     191.294ms      21.316us          8974  
                                aten::cudnn_convolution         0.85%     158.677ms         1.34%     250.914ms     394.519us        1.965s        18.25%        2.128s       3.346ms           636  
                             aten::convolution_backward         0.37%      69.721ms         3.03%     566.855ms     891.281us        4.861s        45.15%        4.861s       7.643ms           636  
                                  cudaFuncGetAttributes         0.20%      37.388ms         0.20%      37.388ms       1.558ms       0.000us         0.00%       0.000us       0.000us            24  

加上pin_memory=true后，profiler如下：
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        41.84%        8.163s        42.31%        8.254s     515.892ms       0.000us         0.00%      12.147ms     759.199us            16  
                                  cudaStreamSynchronize        40.59%        7.919s        40.59%        7.919s     329.948ms       0.000us         0.00%       0.000us       0.000us            24  
                       Runtime Triggered Module Loading         6.76%        1.320s         6.76%        1.320s      18.328ms     132.019ms         1.24%     132.019ms       1.834ms            72  
                                  cudaDeviceSynchronize         3.97%     774.980ms         3.97%     774.980ms     193.745ms       0.000us         0.00%       0.000us       0.000us             4  
                                       cudaLaunchKernel         2.95%     575.628ms         6.99%        1.364s     152.037us       0.000us         0.00%     195.528ms      21.788us          8974  
                                aten::cudnn_convolution         0.95%     184.828ms         1.50%     293.274ms     461.123us        1.952s        18.34%        2.119s       3.332ms           636  
                                          cudaHostAlloc         0.42%      81.392ms         0.42%      81.392ms      13.565ms       0.000us         0.00%       0.000us       0.000us             6  
                             aten::convolution_backward         0.36%      69.681ms         3.06%     597.189ms     938.977us        4.876s        45.82%        4.876s       7.667ms           636  

分析：enumerate时间相比前述减少了4秒左右，但是对应的MemcpyAsync时间也增加了4秒左右，也就是说这两者其实都是瓶颈。因为没开non blocking，虽然pin memory让DMA的准备时间减少（cudaMemcpyAsync，将pageable内存转换为pin memory等），但是因为CPU要求同步，所以转变为了StreamSynchronize，pin memory自身成本反而导致了时间增加。

增加样本校准后（统计所有batch size的和）profiler前几项基本不变，但是enumerate变慢了2s，暂未分析出是为什么，理论上说是因为增加了一次乘法和两次加法。考虑了一下还是调整batch size使所有相等好了。将batch size从128改为96


增加了cudnn.benchmark=True之后，第2、3个epoch的profiler基本没变，但是由于第一个epoch的CPU时间暴增19s（当然，绝大多数是cuda时间），所以就取消了。看来默认的启发式已经相当够用了


将resnet50的训练数据的变换拆成了两部分，第一部分只产生uint8，然后传输到GPU之后再进行normalize来缩小传输时间，产生了大量的StreamSync时间，难道是epoch之间同步也会影响吗？
后：
```
                                               aten::to         0.01%     567.794us        40.22%        4.005s       7.543ms       0.000us         0.00%      14.079ms      26.514us           531
                                         aten::_to_copy         0.00%     414.985us        40.21%        4.005s      88.992ms       0.000us         0.00%      14.079ms     312.866us            45
                                            aten::copy_         0.01%     805.271us        40.20%        4.004s      88.969ms      14.079ms         0.20%      14.079ms     312.866us            45
                                  cudaStreamSynchronize        40.16%        3.999s        40.16%        3.999s     222.187ms       0.000us         0.00%       0.000us       0.000us            18
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        35.62%        3.547s        35.79%        3.564s     296.980ms       0.000us         0.00%       0.000us       0.000us            12
                                  cudaDeviceSynchronize        21.34%        2.125s        21.34%        2.125s     708.330ms       0.000us         0.00%       0.000us       0.000us             3
autograd::engine::evaluate_function: ConvolutionBack...         0.06%       6.069ms         0.60%      59.793ms     125.352us       0.000us         0.00%        3.418s       7.165ms           477
                                       cudaLaunchKernel         0.52%      52.123ms         0.52%      52.123ms       8.203us       0.000us         0.00%       0.000us       0.000us          6354
```
前：
```
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...        31.94%        4.511s        31.99%        4.518s     376.527ms       0.000us         0.00%       0.000us       0.000us            12
                                  cudaDeviceSynchronize        27.52%        3.888s        27.52%        3.888s        1.296s       0.000us         0.00%       0.000us       0.000us             3
                                       cudaLaunchKernel        14.28%        2.016s        23.89%        3.374s     534.106us       0.000us         0.00%        1.297s     205.346us          6318
                                    Command Buffer Full        23.87%        3.372s        23.87%        3.372s       1.476ms        1.297s        16.90%        1.297s     568.027us          2284
autograd::engine::evaluate_function: ConvolutionBack...         0.07%       9.355ms        11.40%        1.610s       3.375ms       0.000us         0.00%        3.644s       7.640ms           477
                                   ConvolutionBackward0         0.02%       2.548ms        10.79%        1.524s       3.195ms       0.000us         0.00%        3.408s       7.144ms           477
                             aten::convolution_backward         0.29%      41.583ms        10.77%        1.522s       3.190ms        3.408s        44.39%        3.408s       7.144ms           477
                                           aten::conv2d         0.01%       1.303ms         3.75%     530.354ms       1.112ms       0.000us         0.00%        2.245s       4.707ms           477
```

取消了epoch之间的print、acc_loss的item以及time()之后，StreamSync的独占CPU时间反而增加了0.8s，连带着其他几个非独占也涨了。那么我们再把profiler关掉对比一下。把profiler关掉之后，取消上述功能的版本4epoch反而慢1.2s，可以撤销了。撤销之后速度也没有提升到之前的，即便把代码完全还原都没用，暂不清楚为何。

将cudaIF恢复加入管线的过程有一些讲究。cudaIF是为了将所有数据预读取、预变换（指CPU变换的部分）并加载到cuda中。原理决定了仅适用于小型数据集。由于是离线变换，所以多线程加载就无用了，num_workers改为0，同时也无需使用pin memory，本来需要将non_blocking关掉的，不过由于cudaIF内部也需要non_blocking参数，所以保持不变。

最终autocast开启后的profiler如下：
                                  cudaDeviceSynchronize        36.50%     999.243ms        36.50%     999.243ms     333.081ms       0.000us         0.00%       0.000us       0.000us             3  
                                           Unrecognized        34.29%     938.786ms        34.29%     938.786ms     440.745us     251.561ms        10.73%     251.561ms     118.104us          2130  
                                       cudaLaunchKernel         8.15%     222.991ms        18.96%     519.117ms     119.173us       0.000us         0.00%     145.096ms      33.310us          4356  
autograd::engine::evaluate_function: ConvolutionBack...         0.29%       7.920ms        12.92%     353.626ms     741.355us       0.000us         0.00%     712.881ms       1.495ms           477  
                                   ConvolutionBackward0         0.06%       1.703ms        11.99%     328.220ms     688.093us       0.000us         0.00%     601.788ms       1.262ms           477  
                             aten::convolution_backward         1.20%      32.840ms        11.93%     326.517ms     684.522us     598.895ms        25.54%     601.788ms       1.262ms           477  
                                           aten::conv2d         0.13%       3.610ms        10.54%     288.519ms     302.431us       0.000us         0.00%     766.769ms     803.741us           954  
                                        cudaMemsetAsync         5.66%     155.043ms         8.46%     231.454ms     135.353us       0.000us         0.00%      43.656ms      25.530us          1710  
                                       aten::batch_norm         0.03%     919.082us         7.18%     196.640ms     412.242us       0.000us         0.00%     368.916ms     773.410us           477  
                           aten::_batch_norm_impl_index         0.07%       2.052ms         7.15%     195.721ms     410.316us       0.000us         0.00%     368.916ms     773.410us           477  
                                 aten::cudnn_batch_norm         0.69%      18.800ms         7.07%     193.669ms     406.014us     281.190ms        11.99%     368.916ms     773.410us           477  

# Neural Network-Based Video Compression Artifact Reduction using Temporal Correlation and Sparsity Prior Predictions(IEEE Access, No.8, 2020). 
Neural Network-Based Video Compression Artifact Reduction using Temporal Correlation and Sparsity Prior Predictions, IEEE Access, DOI: 10.1109/ACCESS.2018.2876864.
# Dependencies
- Keras, TensorFlow, NumPy, Matlab, OpenCV, ...

# Get Started!
## 1. Preparation
(1)使用编码工具(如HM)对视频进行编码, 编码参数可参见sample_parameters.cfg. 目前的设置: I帧的间隔为32, 即第0, 32, 64帧为I帧, 其余为帧; P帧的编码QP=42, I帧的QP=33;
(2)目前只给出了P帧QP = 42的网络参数, 即models/QP42-weights-30.40967.hdf5, 后续会陆续给出QP=37和QP=32;
(3)目录 coded_str 提供了一些编码的视频流;
(4)Useage: python demo.py --video_path  coded_str/Ballpass_D_qp42_dec_65F.yuv --ori_video coded_str/Ballpass_D_ori_65F.yuv --weight_file models/QP42-weights-30.40967.hdf5 --height 240 --width 416
Note: 请按你的实际视频修改 height 和 width参数
(5)输出缺省将保存在 output_A 目录, 一帧对应一幅输出图像, 可通过 --step 参数设置间隔; 图像从左到右依次为未编码的原始图像, 解码图像, 经Compression Artifact reduction 处理的图像;
(6)可通过提供的 comp_psnr_3im.m 计算PSNR和SSIM的增益.

# TODOs
目前的版本没有基于sparse-coding的预测(没有做优化, 比较耗时), 需留待对代码的优化; QP=37, 32的模型参数.

# Yolov5 2_环境配置

## 1. CUDA cuCNN

**存在独立显卡时可使用CUDA和cuCNN进行训练。**

2006年，NVIDIA公司发布了CUDA(Compute Unified Device Architecture)，是一种新的操作GPU计算的硬件和软件架构，是建立在NVIDIA的GPUs上的一个通用并行计算平台和编程模型，它提供了GPU编程的简易接口，基于CUDA编程可以构建基于GPU计算的应用程序，利用GPUs的并行计算引擎来更加高效地解决比较复杂的计算难题。它将GPU视作一个数据并行计算设备，而且无需把这些计算映射到图形API。操作系统的多任务机制可以同时管理CUDA访问GPU和图形程序的运行库，其计算特性支持利用CUDA直观地编写GPU核心程序。

cuDNN是NVIDIACUDA®深度神经网络库，是GPU加速的用于深度神经网络的原语库。cuDNN为标准例程提供了高度优化的实现，例如向前和向后卷积，池化，规范化和激活层。全球的深度学习研究人员和框架开发人员都依赖cuDNN来实现高性能GPU加速。它使他们可以专注于训练神经网络和开发软件应用程序，而不必花时间在底层GPU性能调整上。cuDNN的加快广泛使用的深度学习框架，包括Caffe2，Chainer，Keras，MATLAB，MxNet，PyTorch和TensorFlow。已将cuDNN集成到框架中的NVIDIA优化深度学习框架容器，访问NVIDIA GPU CLOUD了解更多信息并开始使用。

### CUDA 下载和安装

- cuda 安装

  1. 首先安装 NVIDIA 显卡驱动（如果有的话）。

     [NVIDIA GeForce 驱动程序 - N 卡驱动 | NVIDIA](https://www.nvidia.cn/geforce/drivers/)

     ![NULL](./assets/picture_3.jpg)

     选择 Game Ready 驱动程序下载安装即可。

  2. CUDA 安装

     在显卡驱动被正确安装的前提下，终端输入以下指令：

     ```shell
     $ nvidia-smi
     ```

     ![NULL](./assets/picture_4.jpg)

     可以看到最大支持的 CUDA 版本为 12.3。

     进入 CUDA 下载官网： [CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)，选择 CUDA 版本进行下载。

     ![NULL](./assets/picture_5.jpg)

     ![NULL](./assets/picture_6.jpg)

     按照安装程序安装即可。安装结束后输入以下指令检查环境：

     ```shell
     $ nvcc -V
     ```

     ![NULL](./assets/picture_7.jpg)

- cudnn 安装

  进入 cudnn 官网：[cuDNN Archive | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-archive)

  ![NULL](./assets/picture_8.jpg)

  把解压后的bin、include、lib三个文件夹的文件分别拷贝到CUDA安装目录对应的（bin、include、lib）文件夹中即可。


- Python 的 cuda 安装

	yolov5 支持的 pytorch 版本为 1.8.0。在官网中检查支持的 cuda 版本：
	
	![NULL](./assets/picture_9.jpg)
	
	```shell
	# CUDA 11.1
	$ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
	```

## 2. Anaconda

（略）

## 3. Yolov5 环境配置

### 源码下载

[Yolov5 Github源码地址](https://github.com/ultralytics/yolov5)

```shell
$ git clone https://github.com/ultralytics/yolov5
```

### 预训练模型下载

![NULL](./assets/picture_1.jpg)

为了缩短网络的训练时间，并达到更好的精度，一般加载预训练权重进行网络的训练。yolov5提供了以上几个预训练权重，可以对应不同的需求选择不同的版本的预训练权重。在实际场景中是比较看重速度，所以`YOLOv5s`是比较常用的。

### 安装依赖

在yolov5路径下执行：

```shell
$ pip install -r requirements.txt 
```

### 检查安装

- CPU 训练

	在yolov5路径下运行`train.py`文件。
	
	![NULL](./assets/picture_2.jpg)

	出现上述终端日志则成功。
	
- GPU 训练

  在yolov5路径下运行`train.py`文件。修改参数：

  ```python
  parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
  ```

  ![NULL](./assets/picture_10.jpg)

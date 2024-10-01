# PCL-C++ 0_PCL简介和环境配置

> 系列笔记参考资料：
>
> [PCL(Point Cloud Library)学习指南&资料推荐 (yuque.com)](https://www.yuque.com/huangzhongqing/pcl/rdk5k8)

> PCL 官网和 Github：
>
> [官网链接](https://pointclouds.org/)
>
> [Github 链接](https://github.com/PointCloudLibrary/pcl)

## 0. PCL 库简介

- 点云数据的处理可以采用获得广泛应用的**Point Cloud Library (点云库，PCL库)。**
- PCL库是一个最初发布于2013年的开源C++库。它实现了大量点云相关的通用算法和高效的数据管理。
- 支持多种操作系统平台，可在Windows、Linux、Android、Mac OS X、部分嵌入式实时系统上运行。如果说OpenCV是2D信息获取与处理的技术结晶，那么**PCL在3D信息获取与处理上，就与OpenCV具有同等地位**。
- PCL是BSD授权方式，可以免费进行商业和学术应用。

> PCL架构图所示，对于3D点云处理来说，PCL完全是一个的模块化的现代C++模板库。其基于以下第三方库：**Boost、Eigen、FLANN、VTK、CUDA、OpenNI、Qhull**，实现点云相关的**获取、滤波、分割、配准、检索、特征提取、识别、追踪、曲面重建、可视化等。**
>
> ![NULL](./assets/picture_1.jpg)

PCL 学习路线：

![NULL](./assets/picture_2.jpg)

## 1. PCL 环境配置

### Windows 环境配置

> 暂时不使用 Windows 环境学习。
>
> 可以参考：[Windows 配置 PCL 环境](https://blog.csdn.net/xueleiok/article/details/82791629)

### Ubuntu 20.04 环境配置

> 参考教程：[链接](https://blog.csdn.net/qhu1600417010/article/details/120444440)

1. 下载依赖

   ```shell
   $ sudo apt-get update  
   $ sudo apt-get install git build-essential linux-libc-dev
   $ sudo apt-get install cmake cmake-gui
   $ sudo apt-get install libusb-1.0-0-dev libusb-dev libudev-dev
   $ sudo apt-get install mpi-default-dev openmpi-bin openmpi-common 
   $ sudo apt-get install libflann-* libflann-dev
   $ sudo apt-get install libeigen3-dev 
   $ sudo apt-get install libboost-all-dev
   $ sudo apt-get install libvtk7.1-qt libvtk7.1 libvtk7-qt-dev
   $ sudo apt-get install libqhull* libgtest-dev
   $ sudo apt-get install freeglut3-dev pkg-config
   $ sudo apt-get install libxmu-dev libxi-dev
   $ sudo apt-get install mono-complete
   $ sudo apt-get install openjdk-8-jdk openjdk-8-jre
   ```

2. 下载 PCL 源码：https://github.com/PointCloudLibrary/pcl/releases

3. 编译安装 PCL

   ```shell
   $ cd pcl-pcl-1.14.1
   $ mkdir build
   $ cd build
   $ cmake -DCMAKE_BUILD_TYPE=None -DCMAKE_INSTALL_PREFIX=/usr \ -DBUILD_GPU=ON-DBUILD_apps=ON -DBUILD_examples=ON \ -DCMAKE_INSTALL_PREFIX=/usr ..
   $ make
   $ sudo make install
   ```

   

   
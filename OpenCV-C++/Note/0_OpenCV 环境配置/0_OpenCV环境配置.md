# OpenCV C++ 0_环境配置

[OpenCV - Open Computer Vision Library](https://opencv.org/)

## 0. OpenCV 简介

OpenCV（Open Source Computer Vision Library）是一个开源的计算机视觉库。

OpenCV 使用 C/C++ 开发，同时也提供了 Python、Java、MATLAB 等其他语言的接口。

OpenCV 是跨平台的，可以在 Windows、Linux、Mac OS、Android、iOS 等操作系统上运行。

OpenCV 的应用领域非常广泛，包括图像拼接、图像降噪、产品质检、人机交互、人脸识别、动作识别、动作跟踪、无人驾驶等。

OpenCV 还提供了机器学习模块，你可以使用正态贝叶斯、K最近邻、支持向量机、决策树、随机森林、人工神经网络等机器学习算法。

## 1. Windows + C++ 环境配置

~~Windows + MinGW 对 OpenCV 源码编译十分冗杂且失败率高，相比之下使用 MSYS2 构建 OpenCV 库显得简单好用~~

### MSYS2 下载

MSYS2是一组工具和库，为您提供易于使用的环境，用于构建、安装和运行本机Windows软件。为了方便安装软件包并保持更新，它采用了一个名为Pacman的软件包管理系统，ArchLinux用户应该很熟悉。它带来了许多强大的功能，如依赖关系解决和简单的完整系统升级，以及直接和可重现的软件包构建。

[MSYS2](https://www.msys2.org/)

### 使用 Pacman 下载  OpenCV 库和 Eigen 库

```shell
$ pacman -Syu
$ pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-gdb mingw-w64-x86_64-make mingw-w64-x86_64-pkgconf base base-devel msys2-w32api-runtime

$ pacman -Syuu
$ pacman -S mingw-w64-x86_64-gcc
$ pacman -S mingw-w64-x86_64-pkg-config
$ pacman -S mingw-w64-x86_64-zlib

$ pacman -S mingw-w64-x86_64-opencv
$ pacman -S mingw-w64-x86_64-qt6-5c ompat
$ pacman -S mingw-w64-x86_64-vtk
$ pacman -S mingw-w64-x86_64-eigen3
```

### 环境变量配置



## 2. Linux(Ubuntu) + C++ 环境配置


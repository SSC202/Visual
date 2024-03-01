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
$ pacman -S mingw-w64-x86_64-qt6-5compat
$ pacman -S mingw-w64-x86_64-vtk
$ pacman -S mingw-w64-x86_64-eigen3
```

### 环境变量配置

1. 添加系统变量`PKG_CONFIG_PATH`：`...\msys64\mingw64\lib\pkgconfig`；
1. 添加环境变量`PATH`：`...\msys64\mingw64\bin`。

> 比较推荐电脑上仅有这一个`mingw64`环境，以防止多个`mingw64`导致的环境污染（如果有，考虑环境变量顺序）

### VSCode 配置

```json
// c_cpp_properties.json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**",
                "R:/msys64/mingw64/include",
                "R:/msys64/mingw64/include/opencv4"   
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE"
            ],
            "compilerPath": "R:\\msys64\\mingw64\\bin\\g++.exe",                   
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "gcc-x64",
            "configurationProvider": "ms-vscode.cmake-tools"
        }
    ],
    "version": 4
}
```

```json
// launch.json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "g++.exe - 生成和调试活动文件",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}\\${fileBasenameNoExtension}.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "externalConsole": true, //由false改为true后可显示运行框
            "MIMode": "gdb",
            "miDebuggerPath": "R:\\msys64\\mingw64\\bin\\gdb.exe",
            "setupCommands": [
                {
                    "description": "为 gdb 启用整齐打印",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "C/C++: g++.exe 生成活动文件"
        }
    ]
}
```

```json
// tasks.json
{
    "tasks": [
        {
            "type": "cppbuild",
            "label": "C/C++: g++.exe 生成活动文件",
            "command": "R:\\msys64\\mingw64\\bin\\g++.exe",
            "args": [
                "-fdiagnostics-color=always",
                "-std=c++11",
                "-g",
                "${file}",
                "-o",
                "${fileDirname}\\${fileBasenameNoExtension}.exe",
                "-I","R:/msys64/mingw64/include/opencv4", //这里也要改
                "-L","R:/msys64/mingw64/lib", //这里也要改
                "-l","opencv_gapi",
                "-l","opencv_stitching",
                "-l","opencv_alphamat",
                "-l","opencv_aruco",
                "-l","opencv_bgsegm",
                "-l","opencv_ccalib",
                "-l","opencv_cvv",
                "-l","opencv_dnn_objdetect",
                "-l","opencv_dnn_superres",
                "-l","opencv_dpm",
                "-l","opencv_face",
                "-l","opencv_freetype",
                "-l","opencv_fuzzy",
                "-l","opencv_hdf",
                "-l","opencv_hfs",
                "-l","opencv_img_hash",
                "-l","opencv_intensity_transform",
                "-l","opencv_line_descriptor",
                "-l","opencv_mcc",
                "-l","opencv_ovis",
                "-l","opencv_quality",
                "-l","opencv_rapid",
                "-l","opencv_reg",
                "-l","opencv_rgbd",
                "-l","opencv_saliency",
                "-l","opencv_sfm",
                "-l","opencv_stereo",
                "-l","opencv_structured_light",
                "-l","opencv_phase_unwrapping",
                "-l","opencv_superres",
                "-l","opencv_optflow",
                "-l","opencv_surface_matching",
                "-l","opencv_tracking",
                "-l","opencv_highgui",
                "-l","opencv_datasets",
                "-l","opencv_text",
                "-l","opencv_plot",
                "-l","opencv_videostab",
                "-l","opencv_videoio",
                "-l","opencv_viz",
                "-l","opencv_wechat_qrcode",
                "-l","opencv_xfeatures2d",
                "-l","opencv_shape",
                "-l","opencv_ml",
                "-l","opencv_ximgproc",
                "-l","opencv_video",
                "-l","opencv_xobjdetect",
                "-l","opencv_objdetect",
                "-l","opencv_calib3d",
                "-l","opencv_imgcodecs",
                "-l","opencv_features2d",
                "-l","opencv_dnn",
                "-l","opencv_flann",
                "-l","opencv_xphoto",
                "-l","opencv_photo",
                "-l","opencv_imgproc",
                "-l","opencv_core"
            ],
            "options": {
                "cwd": "${fileDirname}"
            },
            "problemMatcher": [
                "$gcc"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "detail": "调试器生成的任务。"
        }
    ],
    "version": "2.0.0"
}
```

## 2. Linux(Ubuntu) + C++ 环境配置


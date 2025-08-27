> 参考资料：[链接](https://blog.csdn.net/qq_36686437/article/details/114160640)

# PCL-C++ 1_PCL数据结构类型

点云是三维空间中，表达目标空间分布和目标表面特性的点的集合，点云通常可以从深度相机中直接获取，也可以从 CAD 等软件中生成。点云是用于表示多维点集合的数据结构，通常用于表示三维数据。在 3D 点云中，这些点通常代表采样表面的X，Y和Z几何坐标。当存在颜色信息时，点云变为 4D 。

三维图像有以下几种表现形式：深度图（描述物体与相机的距离信息），几何模型（由CAD等软件生成），点云模型（逆向工程设备采集生成）

## 1. PCL 文件格式

### PCD 格式

| 文件头      | 声明pcd格式点云数据的某些特性                                |
| ----------- | ------------------------------------------------------------ |
| `VERSON`    | pcd文件的版本信息                                            |
| `FIFLDS`    | 指定的数据点维度和字段名                                     |
| `SIZE`      | 用字节指定的每一维度的大小                                   |
| `TYPE`      | 指定每一维度的数据类型。<br/>目前接受的类型是：<br/>`I` 表示符号类型`int 8`(`Char`)、`int 16`(`Short`)和`int 32`(`Int`)。<br/>`U` 表示无符号类型`uint 8`(`unsigned char`)、`uint 16`(`unsigned short`)、`uint 32`(`unsigned int`)。<br/>`F` 表示浮点类型。 |
| `COUNT`     | 指定每一维度包含的元素数目。<br/>1. 可以为无序点云中的总点数；<br/>2. 可以为有序点云中点云数据的宽度。 |
| `WIDTH`     | 指定的点云集宽度，用点数表示                                 |
| `HEIGHT`    | 指定的点云集高度。<br/>1. 可以为有序点云的行总数；<br/>2. 对于无序点云，设置为1。 |
| `VIEWPOINT` | 指定的点云集获取视点，指定为一组平移和一组四元数。           |
| `POINT`     | 总点数                                                       |
| `DATA`      | 存储点云的文本类型，支持ASCII、二进制和二进制压缩。          |

```pcd
# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F FFF
COUNT 1 1 1 1
WIDTH 213
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 213
DATA ascii
0.93773 0.33763 0 4.2108e+06
0.90805 0.35641 0 4.2108e+06
```

### PLY 格式

| ply                                    | 指明为ply格式的文件            |
| -------------------------------------- | ------------------------------ |
| `format asii 1.0`                      | `ascii/binary`，格式版本号     |
| `comment made by Greg Turk`            | 与所有行一样，指定了注释关键字 |
| `element vertex 8`                     | 定义文件中点的个数             |
| `property float x`                     | 顶点属性                       |
| `property float y`                     | 顶点属性                       |
| `property float z`                     | 顶点属性                       |
| `element face 6`                       | 文件中有6个`face`元素          |
| `property list uchar int vertex_index` | `vertex_indices`是一个整数列表 |
| `end_header`                           | 分割标题的结尾                 |

```ply
ply
format ascii1.0           { ascii/binary, formatversion number }
comment made byGreg Turk  { comments keyword specified,like all lines }
comment thisfile is a cube
element vertex8           { define "vertex"element, 8 of them in file }
property floatx           { vertex contains float"x" coordinate }
property floaty           { y coordinate is also avertex property }
property floatz           { z coordinate, too }
element face6             { there are 6"face" elements in the file }
property listuchar int vertex_index { "vertex_indices" is a list of ints }
end_header                 { delimits the end of theheader }
0 0 0                      { start of vertex list }
0 0 1
0 1 1
0 1 0
1 0 0
1 0 1
1 1 1
1 1 0
4 0 1 2 3                  { start of face list }
4 7 6 5 4
4 0 4 5 1
4 1 5 6 2
4 2 6 7 3
4 3 7 4 0
```

## 2. 点云数据类型

### 点云数据类型

| 类型                       | 含义                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `pcl::PointXYZ`            | 欧氏XYZ坐标的点集结构类型                                    |
| `pcl::Intensity`           | 单通道图像灰度强度的点集结构类型                             |
| `pcl::Intensity8u`         | 单通道图像灰度强度的点集结构类型                             |
| `pcl::Intensity32u`        | 单通道图像灰度强度的点集结构类型                             |
| `pcl::PointXYZI`           | 欧氏XYZ坐标的点集结构和强度值                                |
| `pcl::PointXYZRGBA`        | 欧氏XYZ坐标和RGB颜色的点集结构类型                           |
| `pcl::PointXY`             | 欧式xy坐标的二维点集结构类型                                 |
| `pcl::PointUV`             | 像素图像坐标的2D点集结构类型                                 |
| `pcl::InterestPoint`       | 具有欧氏XYZ坐标和兴趣值的点集结构类型                        |
| `pcl::Normal`              | 法向量坐标和曲面曲率估计的点集结构类型                       |
| `pcl::Axis`                | 用法向量坐标表示轴的点集结构类型                             |
| `pcl::PointNormal`         | 具有欧氏XYZ坐标和法线坐标和表面曲率估计值的点集结构类型      |
| `pcl::PointXYZRGBNormal`   | 具有欧氏XYZ坐标，强度，法线坐标和表面曲率估计的点集结构类型  |
| `pcl::PointXYZLNormal`     | 具有欧氏XYZ坐标，一个标签，法线坐标和表面曲率估计的点集结构类型 |
| `pcl::PointWithRange`      | 具有欧氏XYZ坐标和浮点数的深度信息的点集结构类型              |
| `pcl::PointWithViewpoint`  | 具有欧氏XYZ坐标和视点的点击结构类型                          |
| `pcl::MomentInvariants`    | 表示三个矩是不变量的点集结构类型                             |
| `pcl::PrincipalRadiiRSD`   | 表示使用RSD计算的最小和最大表面半径（以米为单位）的点集结构类型 |
| `pcl::Boundary`            | 表示点是否位于表面边界的点集结构                             |
| `pcl::PrincipalCurvatures` | 表示主曲率及其大小的点集结构                                 |

* `PointXYZ` – 成员变量: `float x, y, z`

  `PointXYZ` 是使用最常见的一个点数据类型，因为它只包含三维 xyz 坐标信息，这三个浮点数附加一个浮点数来满足存储对齐，用户可利用`points[i].data[0]`，或者`points[i].x`访问点的`x`坐标值。

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  ```

- `PointXYZI` – 成员变量: `float x, y, z, intensity`

  `PointXYZI` 是一个简单的 xyz 坐标加 `intensity`（强度） 的 `point` 类型，理想情况下，这四个变量将新建单独一个结构体，并且满足存储对齐。

  然而，由于 `point` 的大部分操作会把`data[4]` 元素设置成 0 或 1（用于变换），不能让 `intensity` 与 xyz 在同一个结构体中，如果这样的话其内容将会被覆盖。

  因此，对于兼容存储对齐，用三个额外的浮点数来填补`intensity`，这样在存储方面效率较低，但是符合存储对齐要求，运行效率较高。

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  union
  {
      struct
      {
          float intensity;
      };
      float data_c[4];
  };
  ```

- `PointXYZRGBA` – 成员变量: `float x, y, z`；`uint32_t rgba`

  除了 rgba 信息被包含在一个整型变量中，其它的和 `PointXYZI` 类似。

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  union
  {
      struct
      {
          uint32_t rgba;
      };
      float data_c[4];
  };
  ```

- `PointXYZRGB` – 成员变量:  `float x, y, z, rgb`

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  union
  {
      struct
      {
          float rgb;
      };
      float data_c[4];
  };
  ```

- `PointXY`– 成员变量: `float x, y`

  简单的二维点云结构。

  ```c++
  struct
  {
      float x;
      float y;
  };
  ```

- `InterestPoint` – 成员变量: `float x, y, z, strength`

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  union
  {
      struct
      {
          float strength;
      };
      float data_c[4];
  };
  ```

- `Normal` – 成员变量: `float normal[3], curvature`

  另一个最常用的数据类型，`Normal` 结构体表示给定点所在样本曲面上的法线方向，以及对应曲率的测量值。

  由于在 PCL中对曲面法线的操作很普遍，还是用第四个元素来占位，这样就兼容SSE和高效计算，曲率不能被存储在同一个结构体中，因为它会被普通的数据操作覆盖掉。

  ```c++
  union
  {
      float data_n[4];
      float normal[3];
      struct
      {
          float normal_x;
          float normal_y;
          float normal_z;
      };
  }
  union
  {
      struct
      {
          float curvature;
      };
      float data_c[4];
  };
  ```

- `PointNormal` – 成员变量: `float x, y, z; float normal[3], curvature`

  `PointNormal` 是存储 xyz 数据的 点云结构体，并且包括采样点对应法线和曲率。

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  union
  {
      float data_n[4];
      float normal[3];
      struct
      {
          float normal_x;
          float normal_y;
          float normal_z;
      };
  };
  union
  {
      struct
      {
          float curvature;
      };
      float data_c[4];
  };
  ```

- `PointXYZRGBNormal` – 成员变量: `float x, y, z, rgb, normal[3], curvature`

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  union
  {
      float data_n[4];
      float normal[3];
      struct
      {
          float normal_x;
          float normal_y;
          float normal_z;
      };
  }
  union
  {
      struct
      {
          float rgb;
          float curvature;
      };
      float data_c[4];
  };
  ```

- `PointXYZINormal` – 成员变量: `float x, y, z, intensity, normal[3], curvature`

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  union
  {
      float data_n[4];
      float normal[3];
      struct
      {
          float normal_x;
          float normal_y;
          float normal_z;
      };
  }
  union
  {
      struct
      {
          float intensity;
          float curvature;
      };
      float data_c[4];
  };
  ```

- `PointWithRange` – 成员变量: `float x, y, z (union with float point[4]), range`

  `PointWithRange` 除了 `range` 包含从所获得的视点到采样点的距离测量值之外，其它与`PointXYZI` 类似。

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  union
  {
      struct
      {
          float range;
      };
      float data_c[4];
  };
  ```

- `PointWithViewpoint`– 成员变量: `float x, y, z, vp_x, vp_y, vp_z`

  `vp_x`、`vp_y`和`vp_z`以三维点表示所获得的视点。

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  union
  {
      struct
      {
          float vp_x;
          float vp_y;
          float vp_z;
      };
      float data_c[4];
  };
  ```

- `MomentInvariants` – 成员变量: `float j1, j2, j3`

  `MomentInvariants` 是一个包含采样曲面上面片的三个不变矩的点云类型，描述面片上质量的分布情况。

  ```c++
  struct
  {
      float j1,j2,j3;
  };
  ```

- `PrincipalRadiiRSD` - `float r_min, r_max`

  `PrincipalRadiiRSD` 是一个包含曲面块上两个 `RSD` 半径的点云类型。

  ```c++
  struct
  {
      float r_min,r_max;
  };
  ```

- `Boundary` – 成员变量:  `uint8_t boundary_point`

  `Boundary` 存储一个点是否位于曲面边界上的点云类型。

  ```c++
  struct
  {
      uint8_t boundary_point;
  };
  ```

- `PrincipalCurvatures` – 成员变量:  `float principal_curvature[3], pc1, pc2`

  `PrincipalCurvatures` 包含给定点主曲率的点云类型。

  ```c++
  struct
  {
      union
      {
          float principal_curvature[3];
          struct
          {
              float principal_curvature_x;
              float principal_curvature_y;
              float principal_curvature_z;
          };
      };
      float pc1;
      float pc2;
  };
  ```

* `PFHSignature125`  – 成员变量: `float pfh[125]`

  `PFHSignature125`包含给定点的`PFH`（点特征直方图）的点云类型。

  ```c++
  struct
  {
      float histogram[125];
  };
  ```

* `FPFHSignature33`  – 成员变量: `float fpfh[33]`

  `FPFHSignature33` 包含给定点的`FPFH`（快速点特征直方图）的点云类型。

  ```c++
  struct
  {
      float histogram[33];
  };
  ```

* `VFHSignature308`  – 成员变量: `float vfh[308]`

  `VFHSignature308`包含给定点 `VFH`（视点特征直方图）的点云类型。

  ```c++
  struct
  {
      float histogram[308];
  };
  ```

* `Narf36`  – 成员变量: `float x, y, z, roll, pitch, yaw; float descriptor[36]`

  `Narf36` 包含给定点 `NARF`（归一化对齐半径特征）的点云类型。

  ```c++
  struct
  {
      float x,y,z,roll,pitch,yaw;
      float descriptor[36];
  };
  ```

* `BorderDescription`  – 成员变量: `int x, y; BorderTraits traits`

  `BorderDescription`包含给定点边界类型的点云类型。

  ```c++
  struct
  {
      int x,y;
      BorderTraitstraits;
  };
  ```

* `IntensityGradient`  – 成员变量: `float gradient[3]`

  `IntensityGradient`包含给定点强度的梯度数据类型。

  ```c++
  struct
  {
      union
      {
          float gradient[3];
          struct
          {
              float gradient_x;
              float gradient_y;
              float gradient_z;
          };
      };
  };
  ```

* `Histogram`  – 成员变量: `float histogram[N]`

  `Histogram`用来存储一般用途的n维直方图。

  ```c++
  template<int N>
  struct Histogram
  {
      float histogram[N];
  };
  ```

* `PointWithScale`  – 成员变量: `float x, y, z, scale`

  scale表示某点用于几何操作的尺度（例如，计算最近邻所用的球体半径，窗口尺寸等等）。

  ```c++
  struct
  {
      union
      {
          float data[4];
          struct
          {
              float x;
              float y;
              float z;
          };
      };
      float scale;
  };
  ```

* `PointSurfel`  – 成员变量: `float x, y, z, normal[3], rgba, radius, confidence, curvature`

  `PointSurfel` 存储 xyz 坐标、曲面法线、rgb 信息、半径、可信度和曲面曲率的点云类型。

  ```c++
  union
  {
      float data[4];
      struct
      {
          float x;
          float y;
          float z;
      };
  };
  union
  {
      float data_n[4];
      float normal[3];
      struct
      {
          float normal_x;
          float normal_y;
          float normal_z;
      };
  };
  union
  {
      struct
      {
          uint32_trgba;
          float radius;
          float confidence;
          float curvature;
      };
      float data_c[4];
  };
  ```

### 点云特征数据类型

| 类型                            | 含义                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| `pcl::PFHSignature125`          | 表示点云的特征直方图（PFH）的点集结构                        |
| `pcl::PFHRGBSignature250`       | 表示颜色特征点特征直方图的点结构（PFHGB）                    |
| `pcl::PPFSignature`             | 用于存储点对特征（PPF）值的点集结构                          |
| `pcl::CPPFSignature`            | 用于存储点对特征（CPPF）值的点集结构                         |
| `pcl::PPFRGBSignature`          | 用于存储点对颜色特征（PPFRGB）值的点集结构                   |
| `pcl::NormalBasedSignature12`   | 表示4-By3的特征矩阵的基于正常的签名的点结构                  |
| `pcl::ShapeContext1980`         | 表示形状上下文的点结构                                       |
| `pcl::UniqueShapeContext1960`   | 表示唯一形状上下文的点结构                                   |
| `pcl::SHOT352`                  | 表示OrienTations直方图（SHOT）的通用标签形状的点集结构       |
| `pcl::SHOT1344`                 | 表示OrienTations直方图（SHOT）的通用签名-形状+颜色           |
| `pcl::_ReferenceFrame`          | 表示点的局部参照系的结构                                     |
| `pcl::FPFHSignature33`          | 表示快速点特征直方图（FPFH）的点结构                         |
| `pcl::VFHSignature308`          | 表示视点特征直方图（VFH）的点结构                            |
| `pcl::GRSDSignature21`          | 表示全局半径的表面描述符（GRSD）的点结构                     |
| `struct pcl::BRISKSignature512` |                                                              |
| `pcl::ESFSignature640`          | 表示形状函数集合的点结构（ESF）                              |
| `pcl::GASDSignature512`         | 表示全局对准的空间分布（GASD）形状描述符的点结构             |
| `pcl::GASDSignature984`         |                                                              |
| `pcl::GASDSignature7992`        | 表示全局对齐空间分布（GASD）形状和颜色描述符的点结构         |
| `pcl::GFPFHSignature16`         | 表示具有16个容器的GFPFH描述符的点结构                        |
| `pcl::Narf36`                   | 表示NARF描述符的点结构                                       |
| `pcl::BorderDescription`        | 用于存储距离图像中的点位于障碍物和背景之间的边界上的结构     |
| `pcl::IntensityGradient`        | 表示XYZ点云强度梯度的点结构                                  |
| `pcl::Histogram< N >`           | 表示N-D直方图的点结构                                        |
| `pcl::PointWithScale`           | 表示三维位置和尺度的点结构                                   |
| `pcl::PointSurfel`              | 具有欧式XYZ坐标、法向坐标、RGBA颜色、半径、置信值和表面曲率估计的面结构 |
| `pcl::PointDEM`                 | 表示数字高程图的点结构                                       |
| `pcl::GradientXY`               | 表示欧氏XYZ坐标和强度值的点结构                              |

## 3. PCL 的文件 IO

```c++
#include <pcl/io/pcd_io.h>
```

### 从 PCD 文件中读取点云

```c++
    /** \brief 从 PCD 文件中读取点云
      * \param[in] file_name PCD 文件名
      * \param[out] 点云数据的引用
      * \ingroup io
      */
template<typename PointT> inline int loadPCDFile (const std::string &file_name, pcl::PointCloud<PointT> &cloud);
```

### 从点云保存 PCD 文件

```c++
  /**
      * \brief 将点云保存为 PCD 文件
      * \param[in] file_name PCD 文件名
      * \param[in] cloud 点云数据
      * \param[in] binary_mode 保存格式, true 为 BIN 格式
      * \ingroup io
      */
template<typename PointT> inline int savePCDFile (const std::string &file_name, const pcl::PointCloud<PointT> &cloud, bool binary_mode = false);
  /**
      * \brief 将点云保存为 PCD 文件(ASCII)
      * \param[in] file_name PCD 文件名
      * \param[in] cloud 点云数据
      * \ingroup io
      */
template<typename PointT> inline int savePCDFileASCII (const std::string &file_name, const pcl::PointCloud<PointT> &cloud);

   /**
      * \brief 将点云保存为 PCD 文件(BIN)
      * \param[in] file_name PCD 文件名
      * \param[in] cloud 点云数据
      * \ingroup io
      */
template<typename PointT> inline int savePCDFileBinary (const std::string &file_name, const pcl::PointCloud<PointT> &cloud);
```

> - **读取与保存PLY文件**
>
> 后缀命名为`.ply`格式文件，常用的点云数据文件。`ply` 文件不仅可以存储点数据，而且可以存储网格数据. 用编辑器打开一个`ply`文件，观察表头，如果表头element face的值为0,则表示该文件为点云文件，如果element face的值为某一正整数N，则表示该文件为网格文件，且包含 N 个网格.所以利用PCL 读取 `ply` 文件，不能一味用`pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PintT>)`来读取。在读取 `ply` 文件时候，首先要分清该文件是点云还是网格类文件。如果是点云文件，则按照一般的点云类去读取即可，如果`ply`文件是网格类，则需要：
>
> ```c++
> #include <pcl/io/ply_io.h>
> 
> pcl::PolygonMesh mesh;
> pcl::io::loadPLYFile("readName.ply", mesh);
> pcl::io::savePLYFile("saveName.ply", mesh);
> ```
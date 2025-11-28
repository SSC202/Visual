# PCL 2_数据结构

## 1. PCL 数据类型

### PCL 点云数据类型

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

### PCL 点云特征数据类型

| 类型                          | 含义                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| `pcl::PFHSignature125`        | 表示点云的特征直方图（PFH）的点集结构                        |
| `pcl::PFHRGBSignature250`     | 表示颜色特征点特征直方图的点结构（PFHGB）                    |
| `pcl::PPFSignature`           | 用于存储点对特征（PPF）值的点集结构                          |
| `pcl::CPPFSignature`          | 用于存储点对特征（CPPF）值的点集结构                         |
| `pcl::PPFRGBSignature`        | 用于存储点对颜色特征（PPFRGB）值的点集结构                   |
| `pcl::NormalBasedSignature12` | 表示4-By3的特征矩阵的基于正常的签名的点结构                  |
| `pcl::ShapeContext1980`       | 表示形状上下文的点结构                                       |
| `pcl::UniqueShapeContext1960` | 表示唯一形状上下文的点结构                                   |
| `pcl::SHOT352`                | 表示OrienTations直方图（SHOT）的通用标签形状的点集结构       |
| `pcl::SHOT1344`               | 表示OrienTations直方图（SHOT）的通用签名-形状+颜色           |
| `pcl::_ReferenceFrame`        | 表示点的局部参照系的结构                                     |
| `pcl::FPFHSignature33`        | 表示快速点特征直方图（FPFH）的点结构                         |
| `pcl::VFHSignature308`        | 表示视点特征直方图（VFH）的点结构                            |
| `pcl::GRSDSignature21`        | 表示全局半径的表面描述符（GRSD）的点结构                     |
| `pcl::ESFSignature640`        | 表示形状函数集合的点结构（ESF）                              |
| `pcl::GASDSignature512`       | 表示全局对准的空间分布（GASD）形状描述符的点结构             |
| `pcl::GASDSignature7992`      | 表示全局对齐空间分布（GASD）形状和颜色描述符的点结构         |
| `pcl::GFPFHSignature16`       | 表示具有16个容器的GFPFH描述符的点结构                        |
| `pcl::Narf36`                 | 表示NARF描述符的点结构                                       |
| `pcl::BorderDescription`      | 用于存储距离图像中的点位于障碍物和背景之间的边界上的结构     |
| `pcl::IntensityGradient`      | 表示XYZ点云强度梯度的点结构                                  |
| `pcl::Histogram<N>`           | 表示N-D直方图的点结构                                        |
| `pcl::PointWithScale`         | 表示三维位置和尺度的点结构                                   |
| `pcl::PointSurfel`            | 具有欧式XYZ坐标、法向坐标、RGBA颜色、半径、置信值和表面曲率估计的面结构 |
| `pcl::PointDEM`               | 表示数字高程图的点结构                                       |
| `pcl::GradientXY`             | 表示欧氏XYZ坐标和强度值的点结构                              |

## 2. PCL 文件格式

PCL 支持 PCD，PLY，OBJ，XYZ，VTK，PNG，TIF 文件。

### PCD 文件格式

PCD 文件由**文件头**和**点云数据本体**构成。

1. **文件头**：包含了描述点云数据的所有必要信息，每一行都以一个关键字开头。

   ```txt
   # .PCD v0.7 - Point Cloud Data file format
   VERSION 0.7
   FIELDS x y z rgb
   SIZE 4 4 4 4
   TYPE F F F F
   COUNT 1 1 1 1
   WIDTH 213
   HEIGHT 1
   VIEWPOINT 0 0 0 1 0 0 0
   POINTS 213
   DATA ascii
   ```

   - `VERSION`：指定 PCD 文件格式的版本号，通常指定为 0.7；

   - `FIELDS`：定义了每个点所包含的字段名称。如 `x y z` 表示坐标；`x y z rgb` 表示坐标和颜色；`x y z normal_x normal_y normal_z` 表示坐标和法向量。

   - `SIZE`：指定每个字段的数据类型大小(以字节为单位)。4 表示 4 字节的 `float` 或 `int`；8 表示 8 字节的 `double`。需要和 `FIELDS` 和 `TYPE` 一一对应。

   - `TYPE`：指定每个字段的数据类型。

     > - `I`：有符号整数(int，char，short，long)
     > - `U`：无符号整数(unsigned int，...)
     > - `F`：浮点数(float，double)

   - `COUNT`：指定每个字段包含的元素数量。对于标量为 1，对于描述子可能大于 1。

   - `WIDTH`：对于无序点云，表示点云中点的总数量；对于有序点云，表示点云矩阵的宽度。

   - `HEIGHT`：对于无序点云，被设置为 1；对于有序点云，表示点云矩阵的高度。

   - `VIEWPOINT`：点云的获取视点，格式为 `tx ty tz qw qx qy qz`，其中 `t*` 是平移向量，`q*` 是旋转四元数。

   - `POINTS`：点云中点的总数量。

   - `DATA`：点云数据的存储类型。

     > - `ascii`：ASCII 文本格式；
     > - `binary`：二进制格式；
     > - `binary_compressed`：压缩的二进制格式。

2. 数据段：紧跟在文件头之后，格式由 `DATA` 字段决定。

   - `DATA ascii`：数据以可读的 ASCII 文本形式存储。每个点占一行，各个字段的值用空格分隔。

     ```txt
     1.0 2.0 3.5 4.2e-5
     -1.0 2.5 3.0 0.0
     ```

   - `DATA binary`：数据以二进制形式存储。程序会直接将从文件读取的二进制块映射到内存中的点云数据结构。
     
- `DATA binary_compressed`：用一种运行长度编码 (RLE) 的压缩算法，特别适用于有序点云。

#### PCD 文件 IO

1. 点云读取

   ```c++
       /** \brief 从 PCD 文件中读取点云
         * \param[in] file_name PCD 文件名
         * \param[out] 点云数据的引用
         * \ingroup io
         */
   template<typename PointT> inline int loadPCDFile (const std::string &file_name, pcl::PointCloud<PointT> &cloud);
   ```

2. 点云保存

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

### PLY 文件格式

PLY 文件由**文件头**和**数据本体**构成。

1. 文件头：使用纯 ASCII 文本，以 `end_header` 行结束。

   ```txt
   ply
   format ascii 1.0
   comment This is a sample PLY file
   element vertex 8
   property float x
   property float y
   property float z
   property uchar red
   property uchar green
   property uchar blue
   element face 6
   property list uchar int vertex_index
   end_header
   ```

   - `ply`：标识为 PLY 文件；

   - `format <format_type> <version>`：定义数据体的存储格式和版本。

     > - `<format_type>` 可以是 `ascii`(可读的 ASCII 文本)；`binary_little_endian`(小端字节序的二进制格式)；`binary_big_endian`(大端字节序的二进制格式)。
     > - `<version>` 通常是 `1.0`。

   - `comment <comment_string>`：注释行；

   - `element <element_name> <count>`：定义一个元素块及其包含的元素数量。通常是 `vertex`(顶点/点)和 `face`(面片)。

     > - `element vertex 8` 表示有 8 个顶点元素，每个顶点有一组属性。
     > - `element face 6` 表示有 6 个面片元素。

   - `property <data_type> <property_name>`：定义一个元素的属性，属性必须紧跟在所属的 `element` 声明之后。

     > - 对于 `vertex` 元素，常见的属性有：
     >   - `property float x`
     >   - `property float y`
     >   - `property float z`
     >   - `property uchar red` (颜色，0-255)
     >   - `property uchar green`
     >   - `property uchar blue`
     >   - `property float nx` (法向量 x 分量)
     >   - `property float ny`
     >   - `property float nz`
     >
     > - `<data_type>` 可以是 `char`，`uchar`，`short`，`ushort`，`int`，`uint`，`float`，`double` 等。

   - `property list <count_type> <index_type> <property_name>`：定义一个列表属性，用于描述面片。

     > - `<count_type>`：表示列表长度的数据类型(通常是 `uchar` 或 `int`)。
     > - `<index_type>`：表示顶点索引的数据类型(通常是 `int`)。
     > - `<vertex_index>`：属性名，表示这个列表存储的是构成面片的顶点索引。

   - `end_header`：标志着文件头结束。

2. 数据体

   - `format ascii`：数据以可读的 ASCII 文本形式存储。每个元素（顶点或面片）占一行。

     ```txt
     // 顶点数据 (8个顶点，每个顶点有 x, y, z, red, green, blue 属性)
     0 0 0 255 0 0
     1 0 0 255 0 0
     1 1 0 0 255 0
     0 1 0 0 255 0
     0 0 1 0 0 255
     1 0 1 0 0 255
     1 1 1 128 128 128
     0 1 1 128 128 128
     // 面片数据 (6个面片，这里是三角面，所以每个列表以'3'开头)
     3 0 1 2    // 一个由顶点0, 1, 2构成的面片
     3 0 2 3    // 一个由顶点0, 2, 3构成的面片
     4 3 7 6 2  // 一个四边形面片(由4个顶点构成，索引为3,7,6,2)
     ...
     ```

   - `format binary_little_endian` 或 `format binary_big_endian`：数据以二进制形式存储。解析器会直接按照文件头定义的属性顺序和数据类型，将二进制块读入内存。这种方式文件体积小，读写速度极快，但无法用文本编辑器阅读。

#### PLY 文件 IO

在读取 `ply` 文件时候，首先要分清该文件是点云还是网格类文件。如果是点云文件，则按照一般的点云类去读取即可，如果`ply`文件是网格类，则需要：

```C++
#include <pcl/io/ply_io.h>

pcl::PolygonMesh mesh;
pcl::io::loadPLYFile("readName.ply", mesh);
pcl::io::savePLYFile("saveName.ply", mesh);
```


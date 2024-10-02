# PCL-C++ 1_PCL基础模块

## 0. 点云的基本概念

点云是三维空间中，表达目标空间分布和目标表面特性的点的集合，点云通常可以从深度相机中直接获取，也可以从 CAD 等软件中生成。点云是用于表示多维点集合的数据结构，通常用于表示三维数据。在 3D 点云中，这些点通常代表采样表面的X，Y和Z几何坐标。当存在颜色信息时，点云变为 4D 。

三维图像有以下几种表现形式：深度图（描述物体与相机的距离信息），几何模型（由CAD等软件生成），点云模型（逆向工程设备采集生成）

## 1. PCL 基本数据类型

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

## 2. 点云的文件存储

### `PCD` 文件

```
    # .PCD v.5 - Point Cloud Data file format
    VERSION .5
    FIELDS x y z
    SIZE 4 4 4
    TYPE F F F
    COUNT 1 1 1
    WIDTH 397
    HEIGHT 1
    POINTS 397
    DATA ascii
    0.0054216 0.11349 0.040749
    -0.0017447 0.11425 0.041273
    -0.010661 0.11338 0.040916
    0.026422 0.11499 0.032623
    0.024545 0.12284 0.024255
    0.034137 0.11316 0.02507
```

1. `VERSION .5` 指定 `PCD` 文件版本；
   
2. `FIELDS x y z` 指定一个点的每一个维度和字段名字，例如
   
> `FIELDS x y z # XYZ data`
>
> `FIELDS x y z rgb # XYZ + colors`
>
> `FIELDS x y z normal_x normal_y normal_z # XYZ + surface normals`

3. `SIZE 4 4 4` 指定每一个维度的字节数大小；

4. `TYPE F F F` 指定每一个维度的类型，`I`表示`int`，`U`表示`uint`，`F`表示浮点；

5. `COUNT 1 1 1` 指定每一个维度包含的元素数，如果没有COUNT，默认都为1；

6. `WIDTH 397` 点云数据集的宽度；

7. `HEIGHT 1` 点云数据集的高度；

8. `VIEWPOINT 0 0 0 1 0 0 0` 指定点云获取的视点和角度，在不同坐标系之间转换时使用（由3个平移+4个四元数构成）；

9. `POINTS 397` 总共的点数；

10. `DATA ascii` 存储点云数据的数据类型，ASCII和binary。

### 其他文件

`PCD` 不是第 一个支持 3D点云数据的文件类型，尤其是计算机图形学和计算几何学领域，已经创建了很多格式来描述任意多边形和激光扫描仪获取的点云。常见的有下面几种格式：

1. `PLY` 是一种多边形文件格式 , 由 Stanford 大学的 Turk 等人设计开发；
2. `STL` 是 3D Systems 公司创建的模型文件格式,主要应用于 CAD 、 CAM领域 ;
3. `OBJ` 是从几何学上定义的文件格式,首先由 Wavefront Technologies 开发;
4. 其他格式

以上所有格式都有其优缺点，因为他们是在不同时期为了满足不同的需求所创建的，那时很多当今流行的传感器和算法都还没有发明。PCL 中 `PCD` 文件格式的正式发布是 0.7 版本。

### 从 `PCD`文件中读取点云

```c++
    /** \brief 从 PCD 文件中读取点云
      * \param[in] file_name PCD 文件名
      * \param[out] 点云数据的引用
      * \ingroup io
      */
template<typename PointT> inline int loadPCDFile (const std::string &file_name, pcl::PointCloud<PointT> &cloud);
```

### 从点云保存 `PCD` 文件

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

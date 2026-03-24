# 图像变换与基于控制点的图像变形实验报告

本报告针对两套图像处理交互系统进行测试与分析：一是基于仿射变换的全局图像几何变换系统；二是基于薄板样条插值（Thin Plate Spline, TPS）的局部控制点图像变形系统。两个系统均采用 Python 构建，并利用 Gradio 提供 Web 交互界面。

## 1. 代码运行说明

### 1.1 环境依赖与安装 (Installation)
运行这两个脚本需要 Python 3.7 或以上版本。在终端或命令行中执行以下命令以安装必要的第三方库：
` ` `bash
pip install gradio opencv-python numpy
` ` `
*(注：为防止代码块嵌套冲突，上方的反引号中加了空格，实际使用时请去掉空格)*

### 1.2 运行脚本 (Running Script)
1. 将第一段代码保存为 `affine_transform.py`。
2. 将第二段代码保存为 `tps_warping.py`。
3. 在终端中运行脚本启动 Web 界面：
   ` ` `
   python affine_transform.py
   ` ` `
   ` ` `
   python tps_warping.py
   ` ` `
4. 运行后，终端会输出一个本地局域网地址，在浏览器中打开该地址即可进行交互测试。

---

## 2. 实验一：全局图像几何变换 (Affine Transform)

此部分对应第一段代码，实现了一个直观的图像基础变换。

### 2.1 输入与输出
* **输入 (Inputs)**:
    * 一张基础图像 (PIL Image 格式)。
    * **Scale**: 缩放比例（0.1 到 2.0）。
    * **Rotation**: 旋转角度（-180° 到 180°）。
    * **Translation X/Y**: 水平与垂直平移量（-300 到 300 像素）。
    * **Flip Horizontal**: 布尔值，是否进行水平翻转。
* **输出 (Outputs)**:
    * 经过组合仿射变换后的目标图像。

### 2.2 测试结果与分析
该系统支持对图像的平移、旋转、缩放和镜像翻转进行实时预览。
* **矩阵乘法组合**: 代码将所有二维空间变换转化为 $3 \times 3$ 的齐次坐标矩阵。其变换顺序严格定义为：**翻转 -> 旋转与缩放 -> 平移**。
* **数学原理**: 组合矩阵公式可表示为 $M_{combined} = M_{trans} \times M_{rot\_scale} \times M_{flip}$。通过将 OpenCV 返回的 $2 \times 3$ 仿射矩阵提取并应用 `cv2.warpAffine`，保证了多种变换能够一次性高效完成，避免了多次处理导致的图像重采样模糊。
* **边界处理**: 代码在处理前对原图进行了白边 Padding 填充，有效防止了图像在旋转或平移时超出画面边界而被裁剪。

---

## 3. 实验二：基于控制点的面部表情变形 (TPS Warping)

此部分对应第二段代码，实现了一个更高级的、非线性的局部图像变形工具。

### 3.1 输入与输出
* **输入 (Inputs)**:
    * 一张基础图像。
    * **交互式点击坐标 (Source & Target Points)**: 用户在图像上交替点击生成源点（蓝色）和目标点（红色）。系统会绘制绿色箭头指示变形方向。
* **输出 (Outputs)**:
    * 经过 TPS 算法扭曲后的结果图像。

### 3.2 测试结果与分析
通过在面部特征（如眼角、嘴角）设置控制点，该系统可以实现类似“微笑”、“闭眼”等面部表情的自然扭曲。
* **逆向映射 (Inverse Warping)**: 为了防止正向映射出现像素“空洞”，系统采用了逆向变形策略。代码以目标点（Destination）作为控制节点，计算出目标图像到源图像的坐标映射关系，最后利用 `cv2.remap` 完成像素插值。
* **薄板样条算法 (TPS)**: 核心数学模型利用了径向基函数 $U(r) = r^2 \log(r^2)$。系统通过求解一个包含正则化项的线性方程组，得到插值权重向量 $W$ 和仿射系数 $A$。正则化参数（代码中设置为 `1e-6`）有效保证了即使在控制点较近时，矩阵求解依然稳定。
* **平滑度**: TPS 算法最小化了映射函数的“弯曲能量”，因此变形结果具有高度的光滑性，非常适合处理生物组织（如人脸）的非刚性形变。

---

## 4. 小结

本实验成功实现了从线性（全局）到非线性（局部）的两套图像变换系统：
1. **基础几何变换**展示了齐次坐标矩阵在图像处理中的强大组合能力，适合做图像的预处理、数据增强（Data Augmentation）等任务。
2. **TPS 控制点变形系统**则展示了高级的空间插值技术，它允许用户进行高度定制化的局部图像编辑，在医疗图像配准、面部动画生成、图像扭曲特效等领域具有重要的工程价值。

---

## 5. 参考论文

关于第二段代码中使用的 Thin Plate Spline (TPS) 算法，其核心理论基础可参考以下经典文献：

* **Bookstein, F. L. (1989).** "Principal warps: Thin-plate splines and the decomposition of deformations." *IEEE Transactions on pattern analysis and machine intelligence*, 11(6), 567-585.
* **Donato, G., & Belongie, S. (2002).** "Approximation methods for thin plate spline mappings and principal warps." *Computer Vision–ECCV 2002: 7th European Conference on Computer Vision*, Copenhagen, Denmark, May 28-31, 2002.
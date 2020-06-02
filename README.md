# gcurval

![License MIT](https://img.shields.io/badge/license-MIT-black) ![CUDA 10.1](https://img.shields.io/badge/CUDA-10.1-brightgreen) ![RMM: RAPIDS Memory Manager](https://img.shields.io/badge/rmm-up%20to%20date-blueviolet) ![Eigen](https://img.shields.io/badge/Eigen-up%20to%20date-ff69b4)


代码对应论文《一种GPU加速的参数曲线弧长计算及弧长采样的方法》，采用CUDA实现，算法原理请参考文章。

这是一个Header-Only的库，使用方式可以参考[samples](./samples)中的两个例子，[椭圆](./samples/sample1_ellipse)和[B样条曲线](./samples/sample2_bspline)，它们分别对应了一个节点区间和多个节点区间的情况。其中[B样条曲线](./samples/sample2_bspline)的例子还展示了如何在算法中使用共享内存作为辅助空间。

# 依赖

- [RMM: RAPIDS Memory Manager](https://github.com/rapidsai/rmm/tree/branch-0.15)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)


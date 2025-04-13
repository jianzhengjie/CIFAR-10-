CIFAR - 10 图像分类三层神经网络
本项目基于 NumPy 构建三层神经网络，实现 CIFAR - 10 图像分类，涵盖数据加载、增强、模型训练、评估、超参数搜索及可视化。
1、依赖项：
  numpy
  pickle
  os
  matplotlib.pyplot
  sklearn
2、文件结构
数据下载文件data_download.py：下载并存放 CIFAR - 10 数据集
主代码文件pic_indenity.py：包含数据处理、模型定义、训练、测试评估等完整逻辑。
3、运行方式
data_download.py下载 CIFAR - 10 数据集至data/目录。
pic_indenity.py会自动进行超参数搜索，训练中实时输出训练轮次、损失及验证集准确率，保存最优模型。训练完成后，输出测试集准确率，并生成训练曲线（损失与验证准确率）、权重可视化图，便于分析模型表现。

# tf_cnnvis

tf_cnnvis是一个CNN可视化库,你可以使用它来更好的理解自己的卷积神经网络。我们使用tesorflow作为后端，生成的图片可以显示在TensorBoard中。目前，我们实现了以下功能：


1.  基于Matthew D. Zeiler and Rob Fergus的论文Visualizing and Understanding Convolutional Networks。我们从卷积神经网络中任意层的信息重构出了输入图像（input image）<这里应该是重构出和输入图像相同大小的图像，如反卷积可视化>。下面是几个例子：

|   |   |   |   |
| :-----------: | :-----------: | :-----------: | :-----------: |
| <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/1.jpg" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/2.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/3.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/4.png" width="196" height="196"> |
| <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/5.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/6.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/7.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/8.png" width="196" height="196"> |
| <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/9.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/10.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/11.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/12.png" width="196" height="196"> |


图1 原始图像和使用tf_cnnvis从Alexnet的maxpool_1 层，maxpool_2 层，以及maxpool_3 层生成的重构图像。

2. 基于谷歌的Deep dream技术CNN可视化，这有一个解释相关技术的博客（404了）。简而言之，Deep dream尝试构建一个能够最大化指定单元输出的激活值的输入图像。下面是一些例子：

|   |   |   |   |
| :-----------: | :-----------: | :-----------: | :-----------: |
| <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Carbonara.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Ibex.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Elephant.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Ostrich.png" width="196" height="196"> |
| Carbonara | Ibex | Elephant | Ostrich |
| <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Cheese burger.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Tennis ball.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Fountain pen.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Clock tower.png" width="196" height="196"> |
| Cheese burger | Tennis ball | Fountain pen | Clock tower |
| <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Cauliflower.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Baby Milk bottle.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Sea lion.png" width="196" height="196"> | <img src="https://github.com/InFoCusp/ui_resources/blob/master/cnnvis_images/Dolphin.png" width="196" height="196"> |
| Cauliflower | Baby Milk bottle | Sea lion | Dolphin |


## tf_cnnvis 依赖

- Tensorflow (>=1.8)

- numpy

- scipy

- h5py

- wget

- Pillow

- six

- scikit-image


 如果您使用pip包管理工具，你可以使用以下命令安装上述依赖：
 `pip install tensorflow numpy scipy h5py wget Pillow six scikit-image `

## 安装tf_cnnvis的命令：

使用git bash克隆本仓库
`#!bash`
`git clone https://github.com/InFoCusp/tf_cnnvis.git`
运行以下命令：
`#!bash`
`sudo pip install setuptools`
`sudo pip install six`
`sudo python setup.py install`
`sudo python setup.py clean`

### 引用

如果你在自己的工作中使用了tf_cnnvis库，请按如下方式引用
 @misc{tf_cnnvis,
    author = {Bhagyesh Vikani, Falak Shah},
    title = {CNN Visualization},
    year = {2017},
    howpublished = {\url{https://github.com/InFoCusp/tf_cnnvis/}},
    doi = {10.5281/zenodo.2594491}
  }

## 应用程序接口(API)

**tf_cnnvis.activation_visualization(graph_or_path, value_feed_dict, input_tensor=None, layers='r', path_logdir='./Log', path_outdir='./Output')**

这个函数用于生成输入图像在卷积网络中指定层的激活值可视化。（可能是直接将特征图可视化了）

#### 参数列表：

- graph_or_path(tf.Graph object or String)(tf.Graph 对象或者字符串对象)- TensorFlow计算图，或者【存储计算图的路径】 是存储卷积神经网络地址的字符串。（其实就是模型，或者直接传入tensorflow计算图，或者传入存储计算图的地址）

- value_feed_dict(dict)-用来evaluate the graph时向网络中传入的占位符的值。形如，dict:{placeholder1: value1,...}。

- input_tensor (tf.tensor object (Default = None)) 输入到模型中的tensorflow 张量，这就是图像输入到模型中的入口。注意：这不是与模型分离的独立张量/占位符。<其实是向卷积神经网络中传入的输入图像，有了输入图像，才有前向传播，才有特征图，才有可视化，反卷积。>

- layers (list or String (Default = 'r')) –

  参数为字符串列表或者字符串，默认值为'r',其中

  - layerName : 从网络中的layerName层重构图像
  - ‘r’ : 从网络中所有的relu层重构图像
  - ‘p’ : 从网络中所有的池化层重构图像
  - ‘c’ : 从网络中所有的卷积层重构图像

- path_outdir (String (Default = "./Output")) – 参数为字符串，默认是"./Output",用来将结果图像存放到磁盘。

- path_logdir (String (Default = "./Log")) –   参数为字符串，默认是"./Log",用来存储用于TensorBoard可视化的日志文件。

####  返回值（Returns）
*  is_success (boolean) 返回值为布尔类型，见名知意。
----

**tf_cnnvis.deconv_visualization(graph_or_path, value_feed_dict, input_tensor=None, layers='r', path_logdir='./Log', path_outdir='./Output')**
这个函数用于从对应输入图像的卷积神经网络的指定层的特征图重构出可视化。
#### 参数列表：
* graph_or_path(tf.Graph object or String)(tf.Graph 对象或者字符串对象)- TensorFlow计算图，或者【存储计算图的路径】 是存储卷积神经网络地址的字符串。<其实就是模型，或者直接传入tensorflow计算图，或者传入存储计算图的地址>
* value_feed_dict(dict)-用来evaluate the graph时向网络中传入的占位符的值。形如，dict:{placeholder1: value1,...}。
* input_tensor (tf.tensor object (Default = None)) 输入到模型中的tensorflow 张量，这就是图像输入到模型中的入口。注意：这不是与模型分离的独立张量/占位符。<其实是向卷积神经网络中传入的输入图像，有了输入图像，才有前向传播，才有特征图，才有可视化，反卷积。>
* layers (list or String (Default = 'r')) –
参数为字符串列表或者字符串，默认值为'r',其中
* layerName : 从网络中的layerName层重构图像
    * ‘r’ : 从网络中所有的relu层重构图像
    * ‘p’ : 从网络中所有的池化层重构图像
    * ‘c’ : 从网络中所有的卷积层重构图像
* path_outdir (String (Default = "./Output")) – 参数为字符串，默认是"./Output",用来将结果图像存放到磁盘。
* path_logdir (String (Default = "./Log")) – 参数为字符串，默认是"./Log",用来存储用于TensorBoard可视化的日志文件。
#### 返回值（Returns）
* is_success (boolean) 返回值为布尔类型，见名知意。
- ---

**tf_cnnvis.deepdream_visualization(graph_or_path, value_feed_dict, layer, classes, input_tensor=None, path_logdir='./Log', path_outdir='./Output')**
这个函数用于从对应输入图像的卷积神经网络的指定层的特征图重构出可视化。
#### 参数列表：
* graph_or_path(tf.Graph object or String)(tf.Graph 对象或者字符串对象)-TensorFlow计算图，或者【存储计算图的路径】 是存储卷积神经网络地址的字符串。<其实就是模型，或者直接传入tensorflow计算图，或者传入存储计算图的地址。>
* value_feed_dict(dict)-用来evaluate the graph时向网络中传入的占位符的值。形如，dict:{placeholder1: value1,...}。
* layer (String) - 字符串参数，TensorFlow计算图中的层的名字。
* classes (List) - 列表参数，为最后一个分类层中特征图索引列表。
* input_tensor (tf.tensor object (Default = None)) 输入到模型中的tensorflow 张量，这就是图像输入到模型中的入口。注意：这不是与模型分离的独立张量/占位符。<其实是向卷积神经网络中传入的输入图像，有了输入图像，才有前向传播，才有特征图，才有可视化，反卷积。>
* path_outdir (String (Default = "./Output")) – 参数为字符串，默认是"./Output",用来将结果图像存放到磁盘。
* path_logdir (String (Default = "./Log")) – 
参数为字符串，默认是"./Log",用来存储用于TensorBoard可视化的日志文件。
#### 返回值（Returns）
* is_success (boolean) 返回值为布尔类型，见名知意。
-----

## 在TensorBoard中可视化
运行以下命令启动Tensorboard：

    #!bash
    tensorboard --logdir=./Log
 然后在TensorBoard主页上查看Images选项卡。
 
 ---

## 额外的辅助函数

**tf_cnnvis.utils.image_normalization(image, ubound=255.0, epsilon=1e-07)**
执行最小-最大图像归一化。将像素强度值转换为range [0, ubound]
#### 参数列表：
* image(3-D numpy array)-一个要归一化的numpy数组。
* ubound (float (Default = 255.0)) -图像像素值的上界。
#### 返回值：
* norm_image (3-D numpy数组)-归一化的图像。
---
**tf_cnnvis.utils.convert_into_grid(Xs, padding=1, ubound=255.0)**
将4-D numpy 数组转换图像网格来显示。
#### 参数列表：
* Xs (4-D numpy array(first axis contains an image))-要放到网格上的图像的4D数组
* padding(int(Default = 1))-网格单元格之间的间距。
* ubound (float (Default = 255.0)) -图像像素值的上界。
#### 返回值：
* (3-D numpy array)-输入图像的网格。
---

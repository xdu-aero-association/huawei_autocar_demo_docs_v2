# ModelArts平台 - 自定义Caffe使用示例

## 1、自定义caffe编译

目前，ModelArts notebook中提供了官方1.0版本的Caffe引擎，但不能直接用于训练无人车需要使用的SSD算法，因此需要另外编译安装caffe-ssd版本的代码。我们可以使用notebook自带的terminal来编译安装caffe-ssd，步骤如下。

### 1.1 创建Notebook

注意要选择Python2、GPU、EVS的配置

![1562657267427](../demo_docs/images/create_notebook.png)

### 1.2 从obs下载caffe代码

我们在OBS上提供了caffe代码，是从https://github.com/weiliu89/caffe/tree/ssd下载的代码。

下载方法有两步：

（1）新建ipynb文件

![1562721244236](../demo_docs/images/notebook.png)

（2）执行以下下载命令

```
# 下载未编译的代码
import moxing as mox
mox.file.copy_parallel('s3://modelarts-competitions/unmanned_vehicle/caffe_demo/compile_caffe_ssd_example/uncompiled/caffe/', 
                       '/home/ma-user/work/caffe')
```

如下图所示：

![1562721190194](../demo_docs/images/copy_caffe.png)



### 1.3 编译caffe

有5个步骤

#### 1.3.1 新建Terminal

![1562721266856](../demo_docs/images/create_terminal.png)

进入caffe目录：

```
cd work/caffe/
```

#### 1.3.2 配置core dump

在Linux系统中，当程序在运行过程中异常终止或崩溃，操作系统会将程序当时的内存状态记录下来，保存在一个“core.xxxx”文件中，这种行为就叫做core dump，有的中文翻译成“核心转储”。

我们可以认为core dump是“内存快照”，但实际上，除了内存信息之外，还有些关键的程序运行状态也会同时dump下来，例如CPU中的寄存器信息、内存管理信息、和操作系统状态等。

core dump对于编程人员诊断和调试程序是非常有帮助的，因为对于有些程序错误是很难重现的，例如指针异常，而core dump文件可以再现程序出错时的情景。但是，如果你的程序在运行时使用的内存比较大，那么core dump保存的core文件就会很大。

在Linux上，是否保存core文件是可以进行配置的，我们所使用的notebook虚拟机默认是打开了core文件的保存，这样可能会引起一个问题：当我们在运行caffe代码出错时，持续地保存core文件很快就可能占满了5G的EVS存储空间，**所以我们推荐关掉notebook虚拟机上的core dump**。

使用如下几条命令可以管理core dump：

```
ulimit -c unlimited  # 打开core dump
ulimit -c 0  # 关闭core dump，执行此条命令
ulimit -c  # 查看core dump的配置值
```

#### 1.3.3 修改Makefile.config文件

拷贝Makefile.config文件：

```
cp Makefile.config.example Makefile.config
```

打开Makefile.config，复制粘贴如下内容，保存：

```
## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# cuDNN acceleration switch (uncomment to build with cuDNN).
USE_CUDNN := 1

# CPU-only switch (uncomment to build without GPU support).
# CPU_ONLY := 1

# uncomment to disable IO dependencies and corresponding data layers
# USE_OPENCV := 0
# USE_LEVELDB := 0
# USE_LMDB := 0

# uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
#	You should not set this flag if you will be reading LMDBs with any
#	possibility of simultaneous read and write
# ALLOW_LMDB_NOLOCK := 1

# Uncomment if you're using OpenCV 3
# OPENCV_VERSION := 3

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
# CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
CUDA_DIR := /usr/local/cuda
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the lines after *_35 for compatibility.
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
             -gencode arch=compute_20,code=sm_21 \
             -gencode arch=compute_30,code=sm_30 \
             -gencode arch=compute_35,code=sm_35 \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_52,code=sm_52 \
             -gencode arch=compute_60,code=sm_60

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := atlas
# BLAS := open
# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

# Homebrew puts openblas in a directory that is not on the standard search path
# BLAS_INCLUDE := $(shell brew --prefix openblas)/include
# BLAS_LIB := $(shell brew --prefix openblas)/lib

# This is required only if you will compile the matlab interface.
# MATLAB directory should contain the mex binary in /bin.
# MATLAB_DIR := /usr/local
# MATLAB_DIR := /Applications/MATLAB_R2012b.app

# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
PYTHON_INCLUDE := /usr/include/python2.7 \
		/home/ma-user/anaconda2/envs/Caffe-1.0.0/lib/python2.7/site-packages/numpy/core/include
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
# ANACONDA_HOME := $(HOME)/anaconda2
# PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		$(ANACONDA_HOME)/include/python2.7 \
		$(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \

# Uncomment to use Python 3 (default is Python 2)
# PYTHON_LIBRARIES := boost_python3 python3.5m
# PYTHON_INCLUDE := /usr/include/python3.5m \
#                 /usr/lib/python3.5/dist-packages/numpy/core/include

# We need to be able to find libpythonX.X.so or .dylib.
PYTHON_LIB := /usr/lib
# PYTHON_LIB := $(ANACONDA_HOME)/lib

# Homebrew installs numpy in a non standard path (keg only)
# PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
# PYTHON_LIB += $(shell brew --prefix numpy)/lib

# Uncomment to support layers written in Python (will link against Python libs)
# WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial

# If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
# INCLUDE_DIRS += $(shell brew --prefix)/include
# LIBRARY_DIRS += $(shell brew --prefix)/lib

# Uncomment to use `pkg-config` to specify OpenCV library paths.
# (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
# USE_PKG_CONFIG := 1

# N.B. both build and distribute dirs are cleared on `make clean`
BUILD_DIR := build
DISTRIBUTE_DIR := distribute

# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
# DEBUG := 1

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0

# enable pretty build (comment to see full commands)
Q ?= @

```

#### 1.3.4 修改Makefile文件

按照如下内容进行修改，箭头左边是修改前，右边是修改后，可以直接复制箭头后的内容覆盖掉原来的：

```
第181行，修改最后两个值：
LIBRARIES += glog gflags protobuf boost_system boost_filesystem boost_regex m hdf5_hl hdf5
===>
LIBRARIES += glog gflags protobuf boost_system boost_filesystem boost_regex m hdf5_serial_hl hdf5_serial
```

#### 1.3.5 编译caffe

```
sudo ln -snf /usr/local/cuda-8.0 /usr/local/cuda
sudo ln -snf  /usr/lib/x86_64-linux-gnu/libcudnn.so.6 /etc/alternatives/libcudnn_so
make all -j16  # 大约需4分钟
make test -j16  # 大约需1分钟
make runtest -j16  # 大约需10分钟，见到下图中的PASSED 2361 tests，则表示测试通过
make pycaffe  # 如果没有特殊的提示信息，则表示编译成功
make distribute  # 如果没有特殊的提示信息，则表示编译成功
```

测试通过的示意图：

![1562660175712](../demo_docs/images/test.png)



## 2、使用caffe训练

### 2.1 数据准备

#### 2.1.1 LMDB数据生成

##### 2.1.1.1 数据采集

**本次demo跳过2.1.1.1和2.1.1.2节，直接执行2.1.1.3节**

准备红绿灯检测训练集图片，整理至同一个本地文件夹。

下载安装[OBS browser](https://support.huaweicloud.com/clientogw-obs/zh-cn_topic_0045829056.html)，使用该软件将训练集图片文件夹上传至自己的OBS桶内，推荐将原始图片放在名称为raw_datasets的目录内。

##### 2.1.1.2 数据标注

进入Modelarts平台的数据标注模块，创建数据集。

其中“数据集输入位置”选择刚才上传的OBS路径，“数据集输出位置”可以自己选一个位置即可，其他选项参考下图进行设置

![1562915436895](../demo_docs/images/create_dataset.png)

标注过程如下：

![1562915506147](../demo_docs/images/annotation.png)

标注结束后，按照下图点击“发布”：

![1562915681647](../demo_docs/images/1562915681647.png)

点击发布后，才会在OBS上的“数据集输出位置”生成新版本的标注文件，有好几级目录，一级一级点进去，会看到一个V001的目录，里面就是最终的xml标签文件。

##### 2.1.1.3 数据集转换

参考1.3.1节，在编译caffe时创建的notebook下打开terminal，执行如下命令，对VOC0712目录做备份：

```
cp -r /home/ma-user/work/caffe/data/VOC0712 /home/ma-user/work/caffe/data/VOC0712_backup
```

上一步标注好的数据存储在OBS路径，我们需要将其拷贝到notebook，参考1.2节的第一步新建ipynb文件，执行如下代码，分别从obs下载图片和xml标签文件：

```
# 下载图片
import moxing as mox
mox.file.copy_parallel('s3://modelarts-competitions/unmanned_vehicle/caffe_demo/compile_caffe_ssd_example/raw_datasets/img_10', 
                       '/home/ma-user/work/caffe/data/VOC0712/traffic_light/JPEGImages')
# 下载标签文件
mox.file.copy_parallel('s3://modelarts-competitions/unmanned_vehicle/caffe_demo/compile_caffe_ssd_example/raw_datasets/img_10_anno', '/home/ma-user/work/caffe/data/VOC0712/traffic_light/Annotations')
```

如果你自己已有标注好的数据，可以将上面的两个s3路径替换为自己的数据所在的路径

在/home/ma-user/work/caffe/data/VOC0712/目录下创建dataset_split.py脚本，粘贴如下代码并保存：

```
# -*- coding: UTF-8 -*-
"""
划分训练验证集和测试集，
划分结果保存为trainval.txt和test.txt，
每个txt文件中的每一行是xml标签文件的前缀，表示该集合中含有该文件
"""
import os 
from sklearn.model_selection import train_test_split

test_percent = 0.25  #测试集占总数据集的比例，可根据任务进行修改

xml_file_path = '/home/ma-user/work/caffe/data/VOC0712/traffic_light/Annotations'  # xml标签文件路径
split_txt_save_path = '/home/ma-user/work/caffe/data/VOC0712/traffic_light/ImageSets/Main'  # 划分结果保存路径
if not os.path.exists(split_txt_save_path):
    os.makedirs(split_txt_save_path)

file_names = os.listdir(xml_file_path)
xml_files = []
for file_name in file_names:
    if file_name.endswith('.xml'):
        xml_files.append(file_name)
    
indexs = list(range(len(xml_files)))
trainval_indexs, test_indexs, trainval_xml_files, test_xml_files = \
    train_test_split(indexs, xml_files, test_size=0.25, random_state=0)

with open(os.path.join(split_txt_save_path, 'trainval.txt'), 'w') as f:
    for file_name in trainval_xml_files:
        file_name_prefix = file_name.rsplit('.xml', 1)[0]
        f.write(file_name_prefix + '\n')

with open(os.path.join(split_txt_save_path, 'test.txt'), 'w') as f:
    for file_name in test_xml_files:
        file_name_prefix = file_name.rsplit('.xml', 1)[0]
        f.write(file_name_prefix + '\n')
        
print('split dataset success, trainval size: %d, test size: %d' % (len(trainval_xml_files), len(test_xml_files)))

```

打开刚才的terminal，执行如下命令：

```
cd /home/ma-user/work/caffe/data/VOC0712/
source activate /home/ma-user/anaconda2/envs/Caffe-1.0.0
python dataset_split.py
```

看到“split dataset success, trainval size: *, test size: *”则表示执行成功，表示数据集划分情况的trainval.txt和test.txt将会保存在/home/ma-user/work/caffe/data/VOC0712/traffic_light/ImageSets/Main目录下

修改/home/ma-user/work/caffe/data/VOC0712/目录下的labelmap_voc.prototxt，替换为如下内容，并保存:

注意，标签0是预留给背景的，不可用于其他标签

```
item {
  name: "background"
  label: 0
  display_name: "background"
}
item {
  name: "green-gx5P"
  label: 1
  display_name: "green-gx5P"
}
item {
  name: "yellow-FiHO"
  label: 2
  display_name: "yellow-FiHO"
}
item {
  name: "red-Ekhs"
  label: 3
  display_name: "red-Ekhs"
}
item {
  name: "off-Zjkp"
  label: 4
  display_name: "off-Zjkp"
}

```

修改/home/ma-user/work/caffe/data/VOC0712/目录下的create_list.sh，替换为如下内容，并保存:

```
#!/bin/bash

root_dir=$HOME/work/caffe/data/VOC0712/  # 修改为数据集目录的上一层目录路径
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in traffic_light  # 修改为数据集目录名
  do
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    img_file=$bash_dir/$dataset"_img.txt"
    cp $dataset_file $img_file
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file

    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file
  done

  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done

```

修改/home/ma-user/work/caffe/data/VOC0712/目录下的create_data.sh，替换为如下内容，并保存:

```
cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=1
data_root_dir="$HOME/work/caffe/data/VOC0712/"  # 修改为数据集目录的上一层目录路径
dataset_name="traffic_light"  # 修改为数据集名称
mapfile="$data_root_dir/labelmap_voc.prototxt"  # 修改为labelmap文件路径
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"  # 只支持jpg格式
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db $data_root_dir/examples/$dataset_name
done

```

打开/home/ma-user/work/caffe/scripts/create_annoset.py，第6行写入如下代码，保存：

```
sys.path.insert(0, '/home/ma-user/work/caffe/python')
```

打开刚才的terminal，执行如下命令：

```
bash create_list.sh
bash create_data.sh
```

执行完成后，/home/ma-user/work/caffe/data/VOC0712/traffic_light目录下会生产lmdb的文件夹，文件夹中为LMDB格式的数据集。

### 2.2 准备训练模型

#### 2.2.1 生成prototxt文件

打开/home/ma-user/work/caffe/examples/ssd/ssd_pascal.py

按照如下内容进行修改，箭头左边是修改前，右边是修改后：

```
1、第1行之后插入如下三行：
import os
import sys
sys.path.insert(0, '/home/ma-user/work/caffe/python/')

2、第80行：
resume_training = True  ===>  resume_training = False

3、第85、87行：
# The database file for training data. Created by data/VOC0712/create_data.sh
train_data = "examples/VOC0712/VOC0712_trainval_lmdb"
# The database file for testing data. Created by data/VOC0712/create_data.sh
test_data = "examples/VOC0712/VOC0712_test_lmdb"
===>
# The database file for training data. Created by data/VOC0712/create_data.sh
train_data = "/home/ma-user/work/caffe/data/VOC0712/traffic_light/lmdb/traffic_light_trainval_lmdb"
# The database file for testing data. Created by data/VOC0712/create_data.sh
test_data = "/home/ma-user/work/caffe/data/VOC0712/traffic_light/lmdb/traffic_light_test_lmdb"

4、第243、245、247、249行：
# Directory which stores the model .prototxt file.
save_dir = "models/VGGNet/VOC0712/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/VGGNet/VOC0712/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "jobs/VGGNet/VOC0712/{}".format(job_name)
# Directory which stores the detection results.
output_result_dir = "{}/data/VOCdevkit/results/VOC2007/{}/Main".format(os.environ['HOME'], job_name)
===>
# Directory which stores the model .prototxt file.
save_dir = "/home/ma-user/work/caffe/data/VOC0712/models/VGGNet/VOC0712/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "/home/ma-user/work/caffe/data/VOC0712/models/VGGNet/VOC0712/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "/home/ma-user/work/caffe/data/VOC0712/jobs/VGGNet/VOC0712/{}".format(job_name)
# Directory which stores the detection results.
output_result_dir = "/home/ma-user/work/caffe/data/VOC0712/results/VOC2007/{}/Main".format(os.environ['HOME'], job_name)

5、第262、264、266、269行：
# Stores the test image names and sizes. Created by data/VOC0712/create_list.sh
name_size_file = "data/VOC0712/test_name_size.txt"
# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
# Stores LabelMapItem.
label_map_file = "data/VOC0712/labelmap_voc.prototxt"

# MultiBoxLoss parameters.
num_classes = 21
===>
# Stores the test image names and sizes. Created by data/VOC0712/create_list.sh
name_size_file = "/home/ma-user/work/caffe/data/VOC0712/test_name_size.txt"
# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
pretrain_model = "*.caffemodel" # your caffemodel path
# Stores LabelMapItem.
label_map_file = "/home/ma-user/work/caffe/data/VOC0712/labelmap_voc.prototxt"

# MultiBoxLoss parameters.
num_classes = 5

6、第335行，改成你使用的GPU id，本示例只用一个GPU：
gpus = "0,1,2,3" ===> gpus = "0"

7、第362行，改成你的测试集数量，本示例只用了3张测试图片：
num_test_image = 4952 ===> num_test_image = 3
```

修改完成后，打开刚才的terminal，执行如下命令：

```
python /home/ma-user/work/caffe/examples/ssd/ssd_pascal.py
```

执行完后会出现“/home/ma-user/work/caffe/data/VOC0712/jobs/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300.sh: line 2: ./build/tools/caffe: No such file or directory”的错误提示，忽略即可，不影响

前往/home/ma-user/work/caffe/data/VOC0712/models/VGGNet/VOC0712/SSD_300x300目录，如果看到生成了4个prototxt文件，则表示执行成功

#### 2.2.2 修改prototxt文件

**train.prototxt：**

修改48行的路径为/cache/data_url/traffic_light_trainval_lmdb

修改135行的路径/cache/data_url/labelmap_voc.prototxt

**test.prototxt：**

修改23行的路径为/cache/data_url/traffic_light_test_lmdb

修改30行的路径为/cache/data_url/labelmap_voc.prototxt

将1641、1644、1645、1667行中的/home/ma-user/work/caffe/data/VOC0712替换为/cache/data_url/

**solver.protxt：**

复制粘贴如下内容，保存：

```
train_net: "/cache/data_url/train.prototxt"
test_net: "/cache/data_url/test.prototxt"
test_iter: 1
test_interval: 10000
base_lr: 0.0010000000475
display: 10
max_iter: 20
lr_policy: "multistep"
gamma: 0.10000000149
momentum: 0.899999976158
weight_decay: 0.000500000023749
snapshot: 10
snapshot_prefix: "/cache/train_url/"
solver_mode: GPU
device_id: 0
debug_info: false
snapshot_after_train: true
test_initialization: false
average_loss: 10
stepvalue: 80000
stepvalue: 100000
stepvalue: 120000
iter_size: 1
type: "SGD"
eval_type: "detection"
ap_version: "11point"
```

solver.protxt中的其他参数在调试模型阶段进行设置，请参考https://bbs.huaweicloud.com/blogs/48accd6db67a11e9b759fa163e330718进行修改。

#### 2.2.3 caffe编译目录打包

打开刚才的terminal，执行如下命令：

```
cd /home/ma-user/work/caffe/
tar -zcvf distribute.tar.gz distribute
```

#### 2.2.4 上传到OBS

打开ipynb文件，执行如下命令，将训练所需的数据和文件上传到OBS：

```
import moxing as mox
mox.file.copy('/home/ma-user/work/caffe/distribute.tar.gz',
                       's3://你的s3路径/distribute.tar.gz')
mox.file.copy_parallel('/home/ma-user/work/caffe/data/VOC0712/traffic_light/lmdb/traffic_light_trainval_lmdb',
                      's3://你的s3路径/traffic_light_trainval_lmdb')
mox.file.copy_parallel('/home/ma-user/work/caffe/data/VOC0712/traffic_light/lmdb/traffic_light_test_lmdb',
                      's3://你的s3路径/traffic_light_test_lmdb')
mox.file.copy('/home/ma-user/work/caffe/data/VOC0712/models/VGGNet/VOC0712/SSD_300x300/train.prototxt',
                       's3://你的s3路径/train.prototxt')
mox.file.copy('/home/ma-user/work/caffe/data/VOC0712/models/VGGNet/VOC0712/SSD_300x300/test.prototxt',
                       's3://你的s3路径/test.prototxt')
mox.file.copy('/home/ma-user/work/caffe/data/VOC0712/models/VGGNet/VOC0712/SSD_300x300/solver.prototxt',
                       's3://你的s3路径/solver.prototxt')
mox.file.copy('/home/ma-user/work/caffe/data/VOC0712/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt',
                       's3://你的s3路径/deploy.prototxt')
mox.file.copy('/home/ma-user/work/caffe/data/VOC0712/labelmap_voc.prototxt',
                       's3://你的s3路径/labelmap_voc.prototxt')
mox.file.copy('/home/ma-user/work/caffe/data/VOC0712/test_name_size.txt',
                       's3://你的s3路径/test_name_size.txt')
```

上传完成后的截图如下所示，总共7个文件和2个文件夹：

![1562722870454](../demo_docs/images/dataset.png)



### 2.3 准备训练脚本

打开notebook，执行如下代码将OBS公共桶上的train_ssd.py保存到notebook，

```
mox.file.copy('s3://modelarts-competitions/unmanned_vehicle/caffe_demo/compile_caffe_ssd_example/src/train_ssd.py',
              '/home/ma-user/work/caffe/data/VOC0712/train_ssd.py')
```

在自己的OBS桶上创建一个src目录，将train_ssd.py上传到该目录下

```
mox.file.copy('/home/ma-user/work/caffe/data/VOC0712/train_ssd.py',
's3://你的s3路径/src/train_ssd.py')
```



### 2.4 创建训练作业

其中，“训练脚本目录”为上文提到的src目录，“训练脚本”为src目录下的train_ssd.py

![1562723386558](../demo_docs/images/train.png)

### 2.5 查看训练日志

![1562723949569](../demo_docs/images/log.png)

如果看到上图中的Iteration和loss，则表示训练正在进行。

如果看到上图中的“Snapshotting ... /cache/train_url/xxx.xxx”则表示训练模型已保存到本地的/cache/train_url路径。

![1562723949569](../demo_docs/images/log2.png)

如果看到上图中的“copy models from /cache/train_url to s3://xxx success”，则表示已经成功将本地/cache/train_url保存的模型拷贝到了OBS路径。**<u>注意，一定要看到这一步输出才表示你的训练成功了！</u>**

## 3、推理预测

打开notebook，执行如下命令，将OBS上训练得到的模型文件xxx.caffemodel拷贝到notebook开发环境中：

```
mox.file.copy('s3://你的OBS路径/你的模型名.caffemodel',
              '/home/ma-user/work/caffe/data/VOC0712/model_snpashots_OBS/你的模型名.caffemodel')
```

在/home/ma-user/work/caffe/data/VOC0712目录下创建inference.py，粘贴如下内容进行保存：

```
# -*- coding: utf-8 -*-
import cv2
import sys
sys.path.insert(0, '/home/ma-user/work/caffe/python/')
import caffe
import numpy as np

model = '/home/ma-user/work/caffe/data/VOC0712/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
weights = '/home/ma-user/work/caffe/data/VOC0712/model_snpashots_OBS/你的模型名.caffemodel'
test_img_path = '/home/ma-user/work/caffe/data/VOC0712/traffic_light/JPEGImages/待测试的图片名.jpg'
keep_top_k = 200  # 与/home/ma-user/work/caffe/examples/ssd_pascal.py中设置的keep_top_k参数保持一致

caffe.set_mode_gpu

net = caffe.Net(model, weights, caffe.TEST)


def preprocess(img):
    img = cv2.resize(img, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img.reshape((1, 3, 300, 300))
    net.blobs['data'].data[...] = img
    net.forward()


def postprocess(img, outputs):
    h = img.shape[0]
    w = img.shape[1]
    output0 = np.reshape(outputs[0], (keep_top_k, 1, 1, 7))
    box = output0[..., 3:7] * np.array([w, h, w, h])

    cls = output0[..., 1]
    conf = output0[..., 1]
    return (box.astype(np.int32), conf, cls)


CLASSES = ('background', 'red', 'green', 'yellow', 'off')
img_raw = cv2.imread(test_img_path)
preprocess(img_raw)
box, conf, cls = postprocess(img_raw, net.blobs['detection_out'].data[0])
for i in range(5):
    print('obj conf:', conf[i])
    p1 = (box[i, ..., 0], box[i, ..., 1])
    p2 = (box[i, ..., 2], box[i, ..., 3])
    cv2.rectangle(img_raw, p1, p2, (0, 255, 0))
    p3 = (max(p1[0], 15), max(p1[1], 15))
    title = '%s:%.2f' % (CLASSES[int(cls[i])], conf[i])
    print(p1, p2, conf[i], title)

```



## 4、模型转换

如需将caffe模型部署至Hilens设备上，则要进行.caffemodel到.om的转换。

首先，准备以下三个文件：

![1562900531619](../demo_docs/images/convert_model.png)

在本地创建aipp.cfg，粘贴如下内容保存，在OBS上新建convert_model目录，将其上传到该目录：

```
aipp_op { 
aipp_mode : static
related_input_rank : 0
input_format : YUV420SP_U8
src_image_size_w : 300
src_image_size_h : 300
csc_switch : true
rbuv_swap_switch : false
matrix_r0c0 : 256
matrix_r0c1 : 454
matrix_r0c2 : 0
matrix_r1c0 : 256
matrix_r1c1 : -88
matrix_r1c2 : -183
matrix_r2c0 : 256
matrix_r2c1 : 0
matrix_r2c2 : 359
input_bias_0 : 0
input_bias_1 : 128
input_bias_2 : 128
min_chn_0 : 104
min_chn_1 : 117
min_chn_2 : 123
var_reci_chn_0 : 1.0
var_reci_chn_1 : 1.0
var_reci_chn_2 : 1.0
}
```

xxx.caffemodel为训练输出的caffe模型文件，在notebook中执行如下命令拷贝到OBS：

```
mox.file.copy('/home/ma-user/work/caffe/data/VOC0712/model_snpashots_OBS/你的模型名.caffemodel',
             's3://你的OBS路径/convert_model/你的模型名.caffemodel')
```

deploy.prototxt需做如下修改，在上传到s3://你的OBS路径/convert_model/目录下

![1562904157843](../demo_docs/images/1562904157843.png)

注：SSDDetectionOutput为昇腾310芯片支持的算子。

将文件上传至obs后，进入Hilens的console页面：

![1562900852439](../demo_docs/images/hilens.png)

创建转换任务：

![1562901047291](../demo_docs/images/create_transfer.png)

转换结果：

![1562901685654](../demo_docs/images/trans-result1.png)

![1562901776038](../demo_docs/images/trans_result2.png)

完成。

## 5、注意事项

（1） 如需使用OBS路径，推荐从[OBS browser](https://support.huaweicloud.com/clientogw-obs/zh-cn_topic_0045829056.html)的地址栏进行复制，如下图所示，并且要把地址最前面的“obs://”改成“s3://”，如果是手动输入路径，推荐按照’s3://{桶名}/{绝对路径}’的格式填写，路径中不要含有‘./’和’../’的相对路径，也不要含有多余的斜杠“/”；

![obs_browser](../demo_docs/images/obs_browser.png)

（2） 创建的notebook默认只有5G的EVS存储空间，很容易用完，请注意及时清理不需要的文件，使用’df –h’命令可查看存储空间的使用情况，如下图所示，空间总大小为4.8G、已用20M、可用4.6G。当然您也可以创建大于5G的EVS notebook，超过5G的部分会收费；

![df-h](../demo_docs/images/df-h.png)

（3） notebook中点击删除按钮删除的东西，仍然会保存在/home/ma-user/work/.Trash-1000中，类似于windows中回收站的作用，如果/home/ma-user/work存储空间不足，在notebook terminal中使用如下命令清空/.Trash-1000目录；

```
cd /home/ma-user/work/.Trash-1000
rm -r files
rm -r info
```

（4）  完成代码调试后，如需跑训练，推荐将notebook中调试好的代码传输到OBS，然后创建训练作业，从OBS加载训练数据和训练代码进行训练。使用训练作业跑训练任务有如下几点好处，这些优点都是notebook不具备的：

​    a)   训练作业有版本管理功能，会记录某次训练使用的训练数据、训练代码、运行参数、训练输出目录，还有训练日志、运行时长、资源占用情况等信息；

​    b)   训练作业完成后，可以创建TensorBoard查看训练情况曲线图，有助于分析下一步应该怎样调试模型；

​    c)   调试模型时，也许一次新的训练就只需要改一个参数，使用训练作业可以在某个版本的训练作业上点击“修改”，然后修改要改的参数即可，其他参数不用动，这样很方便地就创建了一个新的训练作业，而且训练作业有版本溯源功能，能可视化地表示在哪个版本基础上创建了新版本；

​    d)   训练作业运行完即自动停止，不会继续收费，而notebook必须手动停止；

​    e)   训练作业输出的模型会自动保存到OBS，如需导入模型并部署成RESTAPI服务，可以直接从指定的训练作业中加载模型，省去了一步步选择OBS路径的麻烦。

（5）  如果notebook卡死，关掉网页，重新打开即可；

（6）训练过程中涉及到很多路径的手动输入，如果发现没有成功保存模型到OBS路径，首先应该检查的就是路径有没有输入错误。
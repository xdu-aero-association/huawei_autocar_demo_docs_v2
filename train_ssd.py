# -*- coding: utf-8 -*-
import os
import argparse
import moxing as mox


def check_data_url(data_url):
    """
    检查data_url中的内容是否合法
    """
    # 如果训练数据路径不存在，则抛出异常
    if not mox.file.exists(data_url):
        raise Exception("data_url: %s is not exist" % data_url)

    # 如果训练数据路径中，不存在模型训练描述文件solver.prototxt，则抛出异常
    if not mox.file.exists(os.path.join(data_url, 'solver.prototxt')):
        raise Exception("data_url: %s is not contain solver.prototxt" % data_url)

        # 如果训练数据路径中，不存在caffe编译结果压缩包，则抛出异常
    if not mox.file.exists(os.path.join(data_url, 'distribute.tar.gz')):
        raise Exception("data_url: %s is not contain distribute.tar.gz" % data_url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', type=str, default=None,
                        help='the training data path on obs')
    parser.add_argument('--train_url', type=str, default=None,
                        help='the training output saving path on obs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='num of gpus')
    args = parser.parse_args()

    check_data_url(args.data_url)  # 检查args.data_url中的内容是否合法

    obs_data_url = args.data_url  # obs上的训练数据路径
    obs_train_url = args.train_url  # obs上的训练结果保存路径
    local_data_url = '/cache/data_url'  # 本地的训练数据路径（相对OBS而言，我们可以将训练作业虚拟机当做一个本地机器）
    local_train_url = '/cache/train_url'  # 本地的训练结果保存路径
    # 将OBS上的数据拷贝到本地，可以避免频繁读写OBS引起的网络开销
    mox.file.copy_parallel(src_url=obs_data_url, dst_url=local_data_url)
    if not os.path.exists(local_train_url):  # 创建本地训练结果保存目录
        os.makedirs(local_train_url)
    # 解压caffe编译结果压缩包
    cmd = 'cd /cache/data_url && tar -xzf distribute.tar.gz'
    os.system(cmd)

    # 开始模型训练过程
    try:
        # Solver file 
        solver_file = os.path.join(local_data_url, 'solver.prototxt')

        # Pre-trained weights file，
        # 如需加载预训练参数文件，需自行准备放在obs上的训练数据路径中，并在训练命令中加上-weights参数
        weights_name = 'VGG_VOC0712_SSD_300x300_iter_33000.caffemodel'
        weights_file = os.path.join(local_data_url, weights_name)

        # 开始训练    
        cmd = '/cache/data_url/distribute/bin/caffe.bin train -solver {}'.format(
            solver_file)  # + ' -weights {}'.format(weights_file)
        gpus = ','.join(str(id) for id in range(args.num_gpus))
        cmd += ' -gpu {}'.format(gpus)
        print('cmd: ' + cmd)
        os.system(cmd)

        # 训练完成后将保存在本地local_train_url的模型拷贝到obs_train_url
        mox.file.copy_parallel(src_url=local_train_url, dst_url=obs_train_url)
        print('copy models from %s to %s success' % (local_train_url, obs_train_url))
        print("task done")
    except BaseException as err:
        print(err)


if __name__ == "__main__":
    main()

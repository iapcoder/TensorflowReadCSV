# -*- coding: utf-8 -*-

"""
--------------------------------------------------------
# @Version : python3.7
# @Author  : wangTongGen
# @File    : readCSV.py
# @Software: PyCharm
# @Time    : 2019/3/22 22:22
---------------------------------------------------------------------
# @Description: this program shows how to get csv data by Tensorflow.
---------------------------------------------------------------------
"""

import tensorflow as tf
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def csvRead(fileList):
    """
    :param fileList: 文件路径+名字的列表
    :return: 读取的内容
    """

    # 1. 构造文件队列
    file_queue = tf.train.string_input_producer(fileList)

    # 2.构造csv阅读器读取队列数据(一行)
    reader = tf.TextLineReader(skip_header_lines=1)

    key, value = reader.read(file_queue)

    # 3.对每一行内容解码
    # records_defaults:指定每一个样本的每一列的数据，指定默认值[["None"],["1.0"]]
    records = [[1.0],[0.0]]
    example, label = tf.decode_csv(value, record_defaults=records)

    # 4.想要读取多个数据，就需要批处理
    example_batch, label_batch = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)

    return example_batch, label_batch


if __name__ == '__main__':

    # 1.找到文件，放入列表 路径+名字 ->列表当中
    file_name = os.listdir("../datas/")

    file_list = [os.path.join("../datas/", file) for file in file_name]

    # print(file_list)

    example_batch, label_batch = csvRead(file_list)

    # 开启会话
    with tf.Session() as sess:

        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 打印读取的内容
        col1, col2 = sess.run([example_batch, label_batch])

        data = sess.run(tf.stack([col1, col2], axis=1))

        print(data)

        coord.request_stop()
        coord.join(threads)















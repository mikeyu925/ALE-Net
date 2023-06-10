# srcfile 需要复制、移动的文件   
# dstpath 目的地址

from glob import glob
# 导入文件处理相关库
import os, shutil

# 遍历函数
import torch


def read_dirs(f_path,target_path):
    # 获取f_path路径下的所有文件及文件夹
    paths = os.listdir(f_path)
    # 获得目标文件后复制过去的绝对路径
    # target_path = r"D:\target"
    # 判断
    for f_name in paths:
        com_path = f_path + "\\" + f_name
        if os.path.isdir(com_path):  # 如果是一个文件夹
            read_dirs(com_path,target_path)  # 递归调用
        if os.path.isfile:  # 如果是一个文件
            try:
                suffix = com_path.split(".")[1]  # suffix=后缀（获取文件的后缀）
            except Exception as e:
                continue  # 对于没有后缀的文件省略跳过
            try:
                # 可以根据自己需求，修改不同的后缀以获得该类文件
                if suffix == "jpg" or suffix == "JPG":  # 获取jpg文件
                    shutil.move(com_path,target_path)
                elif suffix == "png" or suffix == "PNG":  # 获取png文件
                    shutil.move(com_path,target_path)
                else:
                    continue
            except Exception as e:
                print(e)
                continue


if __name__ == "__main__":
    if False:
        f_path = r"F:\Python\Pycharm\Projects\ganS\PEN-Net\datasets\dtd\images"  # 需要遍历的文件路径
        target_path = r"F:\Python\Pycharm\Projects\ganS\PEN-Net\datasets\dtd\dtdall"
        read_dirs(f_path,target_path)  # 调用函数

    xi = torch.Tensor([[1,2],[3,4]])
    print(xi.shape)
    xi2 = xi*xi
    print(xi2)
    print(torch.sqrt(xi2.sum([0,1],keepdims=True)))



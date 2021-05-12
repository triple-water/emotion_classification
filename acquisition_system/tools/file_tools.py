from pathlib import Path
import os


def write_content_overlap(file_path, file_name, content):
    '''
    覆盖方式写入，路径不存在时创建
    :param file_path:
    :param content:
    :return:
    '''
    if not Path.exists(Path(file_path)):
        Path(file_path).mkdir(parents=True)
    with open(str(Path(Path(file_path) / file_name)), "w")as f:
        f.write(content)


def write_content_add(file_path, file_name, content):
    '''
    追加方式写入，路径不存在时创建
    :param file_path:
    :param content:
    :return:
    '''
    if not Path.exists(Path(file_path)):
        Path(file_path).mkdir(parents=True)
    with open(str(Path(Path(file_path) / file_name)), "a+")as f:
        f.write(content + "\n")


if __name__ == '__main__':
    write_content_add("../acq_data/111", "111.txt", "12222333")

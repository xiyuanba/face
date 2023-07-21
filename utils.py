import netifaces as ni
import os

storage_dir = '/opt/third/.local/etsai'
tmp_images_path = '/opt/third/etsai/tmp'


def get_local_ip():
    # 获取所有接口的信息
    interfaces = ni.interfaces()

    # 优先选择有线网卡 (Ethernet) 和无线网卡 (Wi-Fi) 的 IPv4 地址
    for interface in interfaces:
        if interface.startswith("enp"):
            addresses = ni.ifaddresses(interface)
            if ni.AF_INET in addresses:
                ip_info = addresses[ni.AF_INET][0]
                return ip_info["addr"]

        if interface == "wl":
            addresses = ni.ifaddresses(interface)
            if ni.AF_INET in addresses:
                ip_info = addresses[ni.AF_INET][0]
                return ip_info["addr"]

    # 如果没有找到有线网卡或无线网卡的 IPv4 地址，则返回第一个 IPv4 地址
    for interface in interfaces:
        addresses = ni.ifaddresses(interface)
        if ni.AF_INET in addresses:
            ip_info = addresses[ni.AF_INET][0]
            return ip_info["addr"]

    # 如果没有找到任何 IPv4 地址，则返回回送地址（localhost）
    return "127.0.0.1"


def get_storage_path():
    # 获取数据文件路径
    return storage_dir


def get_images_path():
    # 获取图像文件夹路径
    image_dir = os.path.join(storage_dir, 'images')
    return image_dir


def get_tmp_images_path():
    return tmp_images_path


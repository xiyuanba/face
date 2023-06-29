import netifaces as ni


# 获取本机的IP地址
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


# 测试获取本机的IP地址
print("本机的IP地址是:", get_local_ip())

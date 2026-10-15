def format_params(param_str: str | int) -> str:
    """
    将参数量转为 K / M / B 可读字符串
    :param param_str: 参数字符串/数字，如 '3193909'
    :return: 格式化结果，例 3.19M
    """
    num = int(param_str)
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num}"
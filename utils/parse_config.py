def parse_model_config(path):
    """解析yolo-v3层配置文件并返回模块定义"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # 去掉边缘空白
    module_defs = []
    for line in lines:
        if line.startswith('['):  # 这标志着一个新块的开始
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()  # type被设置为[***]的内容
            if module_defs[-1]['type'] == 'convolutional':  # 如果 *** == convolutional
                module_defs[-1]['batch_normalize'] = 0  # 默认不使用批量归一化
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


#                           module_defs = [
#                               {
# [convolutional]                  'type': 'convolutional',
# batch_normalize=1                'batch_normalize': '1',
# filters=32                       'filters': '32',
# size=3             -->           'size': '3',
# stride=1                         'stride': '1',
# pad=1                            'pad': '1',
# activation=leaky                 'activation': 'leaky'
#                               }
#                           ]

def parse_data_config(path):
    """解析数据配置文件"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


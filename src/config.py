import os
import yaml

# class Config(dict):
#     def __init__(self, config_path):
#         with open(config_path, 'r', encoding='UTF-8') as f:
#             self._yaml = f.read()
#             self._dict = yaml.safe_load(self._yaml)
#             self._dict['PATH'] = os.path.dirname(config_path)
#
#     def __getattr__(self, name):
#         if self._dict.get(name) is not None:
#             return self._dict[name]
#
#         # if DEFAULT_CONFIG.get(name) is not None:
#         #     return DEFAULT_CONFIG[name]
#
#         return None
#
#     def print(self):
#         print('Model configurations:')
#         print('---------------------------------')
#         print(self._yaml)
#         print('')
#         print('---------------------------------')
#         print('')


class Dict(dict): #  see https://www.jb51.net/article/186264.htm
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Config(Dict):
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='UTF-8') as f:
            self.__dict = yaml.safe_load(f.read())

        for k, v in self.__dict.items():
            self[k] = v

        del self['_Config__dict']

    def to_dict(self): # 实例.to_dict() 可以返回一个纯字典
        d = {}
        for k, v in self.items():
            d[k] = v
        return d

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        for k, v in self.items():
            print(k,v)
        print('---------------------------------')
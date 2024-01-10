from enum import Enum


class ImputeMethod(Enum):
    SEMI_GLOBAL = 'semi-global'
    GLOBAL = 'global'
    LOCAL = 'local'
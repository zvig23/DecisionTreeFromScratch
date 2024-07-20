from enum import Enum


class ImputeMethod(Enum):
    SEMI_GLOBAL = 'semi-global'
    GLOBAL = 'global'
    LOCAL = 'local'
    BASELINE = 'default-branch'
    MEAN = 'mean'

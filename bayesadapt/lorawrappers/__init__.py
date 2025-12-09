from .lorawrapper import *
from .viwrapper import *
from .deepensemble import *
from .mcdropout import *
from .blob import *
from .scalabl import *
from .svd import *

cls2name = {
    'ScalablLoraWrapper': 'scalabl',
    'BlobLoraWrapper': 'blob',
    'DeepEnsembleLoraWrapper': 'deepensemble',
    'MCDropoutLoraWrapper': 'mcdropout',
    'VILoraWrapper': 'vi',
    'LoraWrapper': 'lora',
}

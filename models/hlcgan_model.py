import tensorflow as tf
from base.base_model import BaseModel
from layers.convolutional import general_conv2d, resnet_block

class HLCGAN(BaseModel):

    img_height = 256
    img_width = 256
    img_layer =  3
    img_size = img_height * img_width

    batch_size = 1
    pool_size = 50
    nfg = 32
    ndf = 64

    def __init__(self, config):
        super(HLCGAN, self).__init__(config)
        self.build_generator()
        self.init_saver()
        self._ngf = 32
        self._ndf = 64
        self._bath_size = 1
        self._pool_size = 50
        self._img_width = 256
        self._img_height = 256
        self._img_depth = 3


    @property
    def ngf(self):
        return self._ngf

    
    def build_generator(self):
        pass


    def init_saver(self):
        pass
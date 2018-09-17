import tensorflow as tf
from base.base_model import BaseModel

class HLCGAN(BaseModel):

    def __init__(self, config):
        super(HLCGAN, self).__init__(config)
        self.build_model()
        self.init_saver()
        self._ngf = 32
        self._ndf = 64
        self._bath_size = 1
        self._pool_size = 50
        self._img_width = 256
        self._img_height = 256
        self._img_depth = 3

    
    def build_model(self):
        pass


    def init_saver(self):
        pass
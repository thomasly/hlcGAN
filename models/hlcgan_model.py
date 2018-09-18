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
    ngf = 32
    ndf = 64

    def __init__(self, config):
        super(HLCGAN, self).__init__(config)
        self.build_generator()
        self.init_saver()

    
    def build_generator(self):
        pass


    def init_saver(self):
        pass
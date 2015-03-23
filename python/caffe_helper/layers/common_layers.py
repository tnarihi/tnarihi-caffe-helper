import numpy as np
from pycuda.elementwise import ElementwiseKernel

from caffe import Layer
import caffe.pycuda_util as pu


class ReshapeLayer(Layer):

    """Reshape

    You should specify indentical blobs for bottom and top
    """

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.shape_ = param['shape']
        self.bottom_shape_ = bottom[0].data.shape
        assert np.prod(self.bottom_shape_) == np.prod(self.shape_)
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        top[0].reshape(*self.shape_)

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].reshape(*self.bottom_shape_)


class DownSamplingLayer(Layer):

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.factor_ = param['factor']
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        top[0].reshape(
            *(bottom[0].data.shape[:2]
              + (bottom[0].data.shape[2] / self.factor_,
                 bottom[0].data.shape[3] / self.factor_)))

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data[:, :, ::self.factor_, ::self.factor_]

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[:, :, ::self.factor_, ::self.factor_] = top[0].diff


class LogLayer(Layer):

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.offset_ = param['offset']
        self.reshape(bottom, top)
        self.k_log_ = ElementwiseKernel(
            "float *bottom, float *top, float offset",
            "top[i] = log(bottom[i] + offset)", 'elemwise_log')

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        with pu.caffe_cuda_context():
            self.k_log_(
                bottom[0].data_as_pycuda_gpuarray(),
                top[0].data_as_pycuda_gpuarray(),
                np.float32(self.offset_))

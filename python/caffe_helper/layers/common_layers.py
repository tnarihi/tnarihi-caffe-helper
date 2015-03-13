
from caffe import Layer


class ReshapeLayer(Layer):

    """Reshape

    You should specify indentical blobs for bottom and top
    """

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.shape_ = param['shape']
        self.bottom_shape_ = bottom[0].data.shape
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

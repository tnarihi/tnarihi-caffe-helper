from caffe import Layer


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


class Gradient4Layer(Layer):
    """4-connected neighborhood gradient
    bottom: (B, C, H, W)
    top: (B, 2, C, H, W)
    
    Gradients for x-axis are stacked to 0-th index in 2nd axis, gradients for
    y-axis are stacked to 1st index.
    """

    def setup(self, bottom, top):
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        assert len(bottom) == 1
        assert len(top) == 1
        assert len(bottom[0].shape) == 4
        b, c, h, w = bottom[0].shape
        top[0].reshape(b, 2, c, h, w)

    def forward(self, bottom, top):
        top[0].data[:, 0, :, :, :-1] = bottom[0].data[..., :, :-1] - \
            bottom[0].data[..., :, 1:]
        top[0].data[:, 1, :, :-1, :] = bottom[0].data[..., :-1, :] - \
            bottom[0].data[..., 1:, :]

    def backward(self, top, propagate_down, bottom):
        if not propagate_down[0]:
            return
        bottom[0].diff[...] = 0
        bottom[0].diff[..., :, :-1] += top[0].diff[:, 0, :, :, :-1]
        bottom[0].diff[..., :, 1:] -= top[0].diff[:, 0, :, :, :-1]
        bottom[0].diff[..., :-1, :] += top[0].diff[:, 1, :, :-1, :]
        bottom[0].diff[..., 1:, :] -= top[0].diff[:, 1, :, :-1, :]

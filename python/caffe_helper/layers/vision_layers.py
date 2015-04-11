import numpy as np

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


class MorphologyLayer(Layer):

    """Morphological operation using cv2 as a backend.
    Only supports dimension in [1, 3] at axis=1.

    parameters
    ----------

    op : "BLACKHAT", "CROSS", "ELLIPSE", "GRADIENT", "RECT",
    "CLOSE", "DILATE", "ERODE", "OPEN", "TOPHAT" which come from `cv2.MORPH_*`.

    kernel: "4nn" is only supported so far

    """

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.op_ = param['op'].upper()
        kernel_ = param['kernel']
        if kernel_ == '4nn':
            self.kernel_ = np.array(
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        else:
            ValueError
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        assert len(bottom) == 1
        assert len(top) == 1
        assert len(bottom[0].shape) == 4
        assert bottom[0].shape[1] == 3 or bottom[0].shape[1] == 1
        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        import cv2
        for i, img in enumerate(bottom[0].data.transpose(0, 2, 3, 1)):
            imgo = cv2.morphologyEx(
                img, getattr(cv2, "MORPH_" + self.op_), self.kernel_)
            top[0].data[i, ...] = imgo.transpose(2, 0, 1)

    def backward(self, top, propagate_down, bottom):
        raise NotImplementedError(
            "%s does not support backward pass." % self.__class__)

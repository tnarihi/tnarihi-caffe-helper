import numpy as np

import caffe
from caffe.proto import caffe_pb2
from caffe.gradient_check_util import GradientChecker


def test_graident_4_layer():
    ba, c, h, w = (2, 3, 4, 4)
    b = caffe.Blob([ba, c, h, w])
    t = caffe.Blob([])
    bottom = [b]
    top = [t]
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.vision_layers"
    lp.python_param.layer = "Gradient4Layer"
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    rng = np.random.RandomState(313)
    b.data[...] = rng.randn(*b.shape)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    assert top[0].shape == (ba, 2, c, h, w)
    assert np.all(
        top[0].data[:, 0, :, :, :-1] ==
        bottom[0].data[..., :, :-1] - bottom[0].data[..., :, 1:])
    assert np.all(
        top[0].data[:, 1, :, :-1, :] ==
        bottom[0].data[..., :-1, :] - bottom[0].data[..., 1:, :])
    checker = GradientChecker(1e-2, 1e-4)
    checker.check_gradient_exhaustive(
        layer, bottom, top)


import numpy as np

import pytest

import caffe
from caffe.proto import caffe_pb2
from caffe.gradient_check_util import GradientChecker

@pytest.fixture(scope="function")
def blob_inplace_init(blob_4_2322):
    b, _, _, t = blob_4_2322
    t.reshape(*b.shape)
    return [t], [t]

@pytest.fixture(scope="module")
def reshape_layer(blob_4_2322):
    b, _, _, t = blob_4_2322
    t.reshape(*b.shape)
    bottom = [t]
    top = [t]
    reshape = (2, 12)
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.common_layers"
    lp.python_param.layer = "ReshapeLayer"
    lp.python_param.param_str = str({'shape': reshape})
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    return layer

def test_reshape_layer_forward(reshape_layer, blob_inplace_init):
    layer = reshape_layer
    bottom, top = blob_inplace_init
    bak = bottom[0].data.copy()
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    assert bottom[0].shape == top[0].shape
    assert top[0].shape == layer.shape_
    assert np.all(bottom[0].data.flat == bak.flat)

def test_reshape_layer_backward(reshape_layer, blob_inplace_init):
    layer = reshape_layer
    bottom, top = blob_inplace_init
    bak = top[0].diff.copy()
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    layer.Backward(top, [True], bottom)
    assert bak.shape == bottom[0].shape
    assert bak.shape == top[0].shape
    assert np.all(bottom[0].diff.flat == bak.flat)


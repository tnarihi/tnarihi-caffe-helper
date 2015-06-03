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


@pytest.fixture(scope="module",
                params=[(False, False), (False, True),
                        (True, False), (True, True)])
def matrix_mult_layer(request):
    if request.config.getoption('caffe_cpu'):
        raise pytest.skip("MatrixMultLayer requires GPU")
    m1 = caffe.Blob((2, 4, 2))
    m2 = caffe.Blob((2, 2, 3))
    t = caffe.Blob([])
    t1, t2 = request.param
    if t1:
        s = m1.shape
        m1.reshape(s[0], s[2], s[1])
    if t2:
        s = m2.shape
        m2.reshape(s[0], s[2], s[1])
    rng = np.random.RandomState(313)
    m1.data[...] = rng.randn(*m1.shape)
    m2.data[...] = rng.randn(*m2.shape)
    bottom = [m1, m2]
    top = [t]
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.common_layers"
    lp.python_param.layer = "MatrixMultLayer"
    lp.python_param.param_str = str({'t1':  t1, 't2': t2})
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    return layer, bottom, top, request.param


def test_matrix_mult_layer_forward(matrix_mult_layer):
    layer, bottom, top, param = matrix_mult_layer
    m1, m2 = bottom
    m1, m2 = m1.data.copy(), m2.data.copy()
    mo = np.zeros(top[0].shape, np.float32)
    for b in xrange(bottom[0].shape[0]):
        x, y = m1[b], m2[b]
        if param[0]:
            x = x.T
        if param[1]:
            y = y.T
        mo[b][...] = np.dot(x, y)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    assert np.allclose(mo, top[0].data)


def test_matrix_mult_layer_backward(matrix_mult_layer):
    layer, bottom, top, _ = matrix_mult_layer
    checker = GradientChecker(1e-3, 1e-2)
    checker.check_gradient_exhaustive(
        layer, bottom, top)


def test_parameter_layer():
    t = caffe.Blob([])
    bottom = []
    top = [t]
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.common_layers"
    lp.python_param.layer = "ParameterLayer"
    lp.python_param.param_str = str(dict(
        shape=(2, 3, 2, 2),
        filler="lambda shape, rng: rng.randn(*shape) * 0.01"))
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    assert len(layer.blobs) == 1
    assert layer.blobs[0].shape == (2, 3, 2, 2)
    param_copy = layer.blobs[0].data.copy()
    layer.Forward(bottom, top)
    assert np.allclose(top[0].data, param_copy)
    checker = GradientChecker(1e-3, 1e-5)
    checker.check_gradient_exhaustive(
        layer, bottom, top)


def test_reduction_layer_mean(blob_4_2322):
    b, _, _, t = blob_4_2322
    bottom = [b]
    top = [t]
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.common_layers"
    lp.python_param.layer = "ReductionLayer"
    lp.python_param.param_str = str({'axis': 1, 'op': 'mean'})
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    rng = np.random.RandomState(313)
    b.data[...] = rng.randn(*b.shape)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    assert np.all(b.data.mean(layer.axis_).reshape(t.shape) == t.data)
    checker = GradientChecker(1e-2, 1e-4)
    checker.check_gradient_exhaustive(
        layer, bottom, top)


def test_reduction_layer_sum(blob_4_2322):
    b, _, _, t = blob_4_2322
    bottom = [b]
    top = [t]
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.common_layers"
    lp.python_param.layer = "ReductionLayer"
    lp.python_param.param_str = str({'axis': 1, 'op': 'sum'})
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    rng = np.random.RandomState(313)
    b.data[...] = rng.randn(*b.shape)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    assert np.all(b.data.sum(layer.axis_, keepdims=True) == t.data)
    checker = GradientChecker(1e-2, 1e-4)
    checker.check_gradient_exhaustive(
        layer, bottom, top)

def test_slice_by_array_layer(blob_4_2322, tmpdir):
    path_indexes = tmpdir.join('indexes.mat').strpath
    from scipy.io import savemat
    indexes = np.array([2, 0])
    savemat(path_indexes, {'indexes': indexes})
    b, _, _, t = blob_4_2322
    bottom = [b]
    top = [t]
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.common_layers"
    lp.python_param.layer = "SliceByArrayLayer"
    lp.python_param.param_str = str(
        {'path_mat': path_indexes, 'key': 'indexes'})
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    rng = np.random.RandomState(313)
    b.data[...] = rng.randn(*b.shape)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    assert np.all(top[0].data == bottom[0].data[:, indexes, ...])
    checker = GradientChecker(1e-2, 1e-5)
    checker.check_gradient_exhaustive(
        layer, bottom, top)


def test_broadcast_layer():
    ba, c, h, w = [2, 1, 3, 4]
    b, t = caffe.Blob([ba, c, h, w]), caffe.Blob([])
    bottom = [b]
    top = [t]
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.common_layers"
    lp.python_param.layer = "BroadcastLayer"
    lp.python_param.param_str = str(
        {'axis': 1, 'num': 3})
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    rng = np.random.RandomState(313)
    b.data[...] = rng.randn(*b.shape)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    assert t.shape == (ba, 3, h, w)
    for i in xrange(3):
        assert np.all(b.data == t.data[:, i:i + 1])
    checker = GradientChecker(1e-2, 1e-5)
    checker.check_gradient_exhaustive(
        layer, bottom, top)


def test_tile_layer():
    ba, c, h, w = [2, 3, 3, 4]
    b, t = caffe.Blob([ba, c, h, w]), caffe.Blob([])
    bottom = [b]
    top = [t]
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.common_layers"
    lp.python_param.layer = "TileLayer"
    axis = 1
    num = 5
    lp.python_param.param_str = str(
        {'axis': axis, 'num': num})
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    rng = np.random.RandomState(313)
    b.data[...] = rng.randn(*b.shape)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    assert t.shape == (ba, c * num, h, w)
    reps = [1 for _ in t.shape]
    reps[axis] = num
    assert np.all(np.tile(b.data, reps) == t.data)
    checker = GradientChecker(1e-2, 1e-5)
    checker.check_gradient_exhaustive(
        layer, bottom, top)


def test_axpb_layer(blob_4_2322):
    b, _, _, t = blob_4_2322
    bottom = [b]
    top = [t]
    # Create Layer
    va = 0.7
    vb = -0.3
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.common_layers"
    lp.python_param.layer = "AXPBLayer"
    lp.python_param.param_str = str({'a': va, 'b': vb})
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    rng = np.random.RandomState(313)
    b.data[...] = rng.randn(*b.shape)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    assert np.all(va * b.data + vb == t.data)
    checker = GradientChecker(1e-3, 1e-2)
    checker.check_gradient_exhaustive(
        layer, bottom, top)

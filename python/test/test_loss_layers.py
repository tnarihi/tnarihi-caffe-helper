import numpy as np

import pytest

import caffe
from caffe.proto import caffe_pb2
from caffe.gradient_check_util import GradientChecker

@pytest.fixture
def sil2_loss_layer(request):
    # Check caffe mode
    was_cpu = caffe.check_mode_cpu
    if was_cpu:
        caffe.set_mode_gpu()
    # Create blobs
    shape = [2, 3, 2, 2]
    pred = caffe.Blob(shape)
    label = caffe.Blob(shape)
    mask = caffe.Blob(shape)
    rng = np.random.RandomState(313)
    pred.data[...] = rng.rand(*shape) + 0.01  # > 0
    label.data[...] = rng.rand(*shape) + 0.01  # > 0
    mask.data[...] = rng.rand(*shape) > 0.2  # 80% and avoid 0 div
    mask.data[mask.data.reshape(mask.shape[0], -1).sum(1) == 0, 0, 0, 0] = 1
    bottom = [pred, label, mask]
    top = [caffe.Blob([])]
    lam = 0.5
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.loss_layers"
    lp.python_param.layer = "ScaleInvariantL2LossLayer"
    lp.python_param.param_str = str({'lambda': lam})
    layer = caffe.create_layer(lp)
    # Finalizer
    def fin():
        if was_cpu:
            caffe.set_mode_cpu()
    request.addfinalizer(fin)
    return layer, bottom, top

def test_sil2_loss_layer_forward(sil2_loss_layer):
    layer, bottom, top = sil2_loss_layer
    pred, label, mask = bottom

    layer.SetUp(bottom, top)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    pred_data = pred.data.reshape(pred.shape[0], -1)
    label_data = label.data.reshape(pred.shape[0], -1)
    mask_data = mask.data.reshape(pred.shape[0], -1)
    diff = (pred_data - label_data) * mask_data
    diff_sum = diff.sum(1)
    diff2_sum = (diff**2).sum(1)
    mask_sum = mask_data.sum(1)
    term1 = (diff2_sum / mask_sum).mean()
    term2 = ((diff_sum ** 2) / (mask_sum ** 2)).mean()
    loss = term1 - layer.lambda_ * term2
    assert np.isclose(top[0].data, loss)

def test_sil2_loss_layer_backward(sil2_loss_layer):
    layer, bottom, top = sil2_loss_layer
    layer.SetUp(bottom, top)
    layer.Reshape(bottom, top)
    checker = GradientChecker(1e-2, 1e-2)
    checker.check_gradient_exhaustive(
        layer, bottom, top, check_bottom=[0, 1])

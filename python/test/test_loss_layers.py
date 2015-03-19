import numpy as np

import pytest

import caffe
from caffe.proto import caffe_pb2
from caffe.gradient_check_util import GradientChecker


@pytest.fixture(scope="module")
def sil2_loss_layer(request, blob_4_2322):
    print "Call:", request.fixturename
    # Create blobs
    pred, label, mask, top = blob_4_2322
    bottom = [pred, label, mask]
    top = [top]
    lam = 0.5
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.loss_layers"
    lp.python_param.layer = "ScaleInvariantL2LossLayer"
    lp.python_param.param_str = str({'lambda': lam})
    # caffe.set_mode_gpu()
    # caffe.set_device(0)
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    return layer


def test_sil2_loss_layer_forward(sil2_loss_layer, blob_4_2322_init):
    layer = sil2_loss_layer
    pred, label, mask, top = blob_4_2322_init
    bottom = [pred, label, mask]
    top = [top]
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    pred_data = pred.data.reshape(pred.shape[0], -1)
    label_data = label.data.reshape(pred.shape[0], -1)
    mask_data = mask.data.reshape(pred.shape[0], -1)
    diff = (pred_data - label_data) * mask_data
    diff_sum = diff.sum(1)
    diff2_sum = (diff ** 2).sum(1)
    mask_sum = mask_data.sum(1)
    term1 = (diff2_sum / mask_sum).mean()
    term2 = ((diff_sum ** 2) / (mask_sum ** 2)).mean()
    loss = term1 - layer.lambda_ * term2
    assert np.isclose(top[0].data, loss)


def test_sil2_loss_layer_backward(sil2_loss_layer, blob_4_2322_init):
    layer = sil2_loss_layer
    pred, label, mask, top = blob_4_2322_init
    bottom = [pred, label, mask]
    top = [top]
    checker = GradientChecker(1e-2, 1e-2)
    checker.check_gradient_exhaustive(
        layer, bottom, top, check_bottom=[0, 1])

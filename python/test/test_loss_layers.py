import numpy as np

import pytest

import caffe
from caffe.proto import caffe_pb2
from caffe.gradient_check_util import GradientChecker


@pytest.fixture(scope="module")
def sil2_loss_layer(request, blob_4_2322):
    if request.config.getoption('caffe_cpu'):
        raise pytest.skip("ScaleInvariantL2LossLayer requires GPU")
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


def test_dssim_layer(request):
    if request.config.getoption('caffe_cpu'):
        raise pytest.skip("DSSIMLayer requires GPU")
    x, y = np.ogrid[:5, :5]
    img1 = np.sin(x / 5.0 * np.pi) * np.cos(y / 5.0 * np.pi)
    img1 = np.repeat(img1[..., np.newaxis], 3, 2)
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    rng = np.random.RandomState(313)
    img2 = img1 + rng.randn(*img1.shape) * 0.2
    img2[img2 > 1] = 1
    img2[img2 < 0] = 0
    bottom = [caffe.Blob([]), caffe.Blob([])]
    top = [caffe.Blob([])]
    img1 = img1.transpose(2, 0, 1)
    img2 = img2.transpose(2, 0, 1)
    bottom[0].reshape(*((1,) + img1.shape))
    bottom[1].reshape(*((1,) + img2.shape))
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.loss_layers"
    lp.python_param.layer = "DSSIMLayer"
    lp.python_param.param_str = str({'hsize': 3})
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    bottom[0].data[...] = img1[np.newaxis]
    bottom[1].data[...] = img2[np.newaxis]
    checker = GradientChecker(1e-3, 1e-2)
    checker.check_gradient_exhaustive(
        layer, bottom, top, check_bottom=[0, 1])


def test_logit_loss_layer(request):
    if request.config.getoption('caffe_cpu'):
        raise pytest.skip("LogitLossLayer requires GPU")
    # data
    y = np.random.rand(2, 3, 4, 5)
    t = (np.random.rand(2, 3, 4, 5) > 0.5).astype(np.float)\
        - (np.random.rand(2, 3, 4, 5) > 0.5).astype(np.float)
    # setting up blobs
    bottom = [caffe.Blob([]), caffe.Blob([])]
    top = [caffe.Blob([])]
    bottom[0].reshape(*y.shape)
    bottom[1].reshape(*t.shape)
    bottom[0].data[...] = y
    bottom[1].data[...] = t
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.loss_layers"
    lp.python_param.layer = "LogitLossLayer"
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    # reference computation
    l = np.sum(np.abs(t) * np.log(1 + np.exp(-y * t))) / np.sum(np.abs(t))
    assert np.isclose(top[0].data, l)
    checker = GradientChecker(1e-3, 1e-2)
    checker.check_gradient_exhaustive(
        layer, bottom, top, check_bottom=[0])


def test_crossent_layer(request):
    if request.config.getoption('caffe_cpu'):
        raise pytest.skip("crossentropyLossLayer requires GPU")
    pred = caffe.Blob((5, 8))
    label = caffe.Blob((5, 1))
    loss = caffe.Blob([])
    bottom = [pred, label]
    top = [loss]

    # Fill
    rng = np.random.RandomState(313)
    pred.data[...] = rng.rand(*pred.shape) + 0.1
    label.data[...] = rng.randint(0, 8, label.shape)
    pred.data[...] = pred.data / pred.data.sum(axis=1, keepdims=True)
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.loss_layers"
    lp.python_param.layer = "CrossEntropyLossLayer"
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    layer.Reshape(bottom, top)
    layer.Forward(bottom, top)
    ref = -np.mean(
        np.log(
            np.maximum(np.finfo(np.float32).tiny, pred.data)
        ).reshape(pred.shape[0], -1)[
            np.arange(pred.shape[0]), label.data.astype('int32')]
    )
    assert np.isclose(ref, loss.data)
    checker = GradientChecker(1e-3, 1e-2)
    checker.check_gradient_exhaustive(
        layer, bottom, top, check_bottom=[0])

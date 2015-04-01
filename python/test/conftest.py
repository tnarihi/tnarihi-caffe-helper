import pytest

import numpy as np
import caffe


def pytest_addoption(parser):
    parser.addoption("--caffe-cpu", action="store_true",
                     dest='caffe_cpu',
                     help="Use cpu instead of gpu")
    parser.addoption("--caffe-gpu", action="store", default=0,
                     dest='caffe_gpu', type='int',
                     help="Specify gpu device id")


def get_name(request):
    """
    try:
            name = request.function.__name__
    except:
            try:
                    name = request.cls.__name__
            except:
                    try:
                            name = request.module.__name__
                    except:
                            name = 'session'
    return name
    """
    return request.fixturename


def set_caffe_dev(request):
    name = get_name(request)
    if request.config.getoption('caffe_cpu'):
        caffe.set_mode_cpu()
        print '"%s" run in cpu' % name
        return
    device_id = request.config.getoption('caffe_gpu')
    caffe.set_mode_gpu()
    caffe.set_device(device_id)
    print '"%s" run in gpu %d' % (name, device_id)

"""
@pytest.fixture(scope="session", autouse=True)
def session_dev(request):
    set_caffe_dev(request)


@pytest.fixture(scope="module", autouse=True)
def module_dev(request):
    set_caffe_dev(request)


@pytest.fixture(scope="class", autouse=True)
def class_dev(request):
    set_caffe_dev(request)
"""


@pytest.fixture(scope="function", autouse=True)
def function_dev(request):
    set_caffe_dev(request)


# Blob FloatingPointError
@pytest.fixture(scope="session")
def blob_4_2322(request):
    print "Call:", request.fixturename
    shape = [2, 3, 2, 2]
    return [caffe.Blob(shape) for i in xrange(4)]


@pytest.fixture(scope="function")
def blob_4_2322_init(request, blob_4_2322):
    print "Call:", request.fixturename
    pred, label, mask, top = blob_4_2322
    shape = pred.shape
    rng = np.random.RandomState(313)
    pred.data[...] = rng.rand(*shape) + 0.01  # > 0
    label.data[...] = rng.rand(*shape) + 0.01  # > 0
    mask.data[...] = rng.rand(*shape) > 0.2  # 80% and avoid 0 div
    mask.data[mask.data.reshape(mask.shape[0], -1).sum(1) == 0, 0, 0, 0] = 1
    return pred, label, mask, top

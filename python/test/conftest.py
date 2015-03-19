import pytest
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
    if request.config.option.caffe_cpu:
        caffe.set_mode_cpu()
        print '"%s" run in cpu' % name
        return
    device_id = request.config.option.caffe_gpu
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

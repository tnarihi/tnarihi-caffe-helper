import numpy as np
from pycuda.elementwise import ElementwiseKernel

import caffe
from caffe import Layer
import caffe.pycuda_util as pu


class ReshapeLayer(Layer):

    """Reshape

    You should specify indentical blobs for bottom and top
    """

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.shape_ = param['shape']
        self.bottom_shape_ = bottom[0].data.shape
        assert np.prod(self.bottom_shape_) == np.prod(self.shape_)
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        top[0].reshape(*self.shape_)

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].reshape(*self.bottom_shape_)


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


class LogLayer(Layer):

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.offset_ = param['offset']
        self.reshape(bottom, top)
        self.k_log_ = ElementwiseKernel(
            "float *bottom, float *top, float offset",
            "top[i] = log(bottom[i] + offset)", 'elemwise_log')

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        with pu.caffe_cuda_context():
            self.k_log_(
                bottom[0].data_as_pycuda_gpuarray(),
                top[0].data_as_pycuda_gpuarray(),
                np.float32(self.offset_))


def blas_trans(t):
    return 'T' if t else 'N'


class MatrixMultLayer(Layer):

    def _check_shape(self, bottom, top):
        assert len(bottom) == 2
        assert len(top) == 1
        assert bottom[0].shape[0] == bottom[1].shape[0]
        r1, c1 = bottom[0].shape[1:]
        r2, c2 = bottom[1].shape[1:]
        if self.t1_:
            r1, c1 = c1, r1
        if self.t2_:
            r2, c2 = c2, r2
        assert c1 == r2
        self.outshape_ = r1, c2

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.t1_ = param.get('t1', False)
        self.t2_ = param.get('t2', False)
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        self._check_shape(bottom, top)
        batch_size = bottom[0].shape[0]
        shape = (batch_size,) + self.outshape_
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        with pu.caffe_cuda_context():
            h = caffe.cublas_handle()
            import scikits.cuda.linalg as linalg
            mat1 = bottom[0].data_as_pycuda_gpuarray()
            mat2 = bottom[1].data_as_pycuda_gpuarray()
            mato = top[0].data_as_pycuda_gpuarray()
            for b in xrange(bottom[0].shape[0]):
                linalg.dot(mat1[b], mat2[b],
                           transa=blas_trans(self.t1_),
                           transb=blas_trans(self.t2_),
                           handle=h, out=mato[b])

    def backward(self, top, propagate_down, bottom):
        with pu.caffe_cuda_context():
            h = caffe.cublas_handle()
            import scikits.cuda.linalg as linalg
            top_diff = top[0].diff_as_pycuda_gpuarray()
            ts = [self.t1_, self.t2_]
            for i in xrange(len(bottom)):
                if not propagate_down[i]:
                    continue
                diff = bottom[i].diff_as_pycuda_gpuarray()
                data = bottom[(i + 1) % 2].data_as_pycuda_gpuarray()
                # Belew 3 conditions are complicated and might be hard to
                # understand.
                swap = ts[i] ^ bool(i)
                t1 = ts[i]
                t2 = (not t1) ^ ts[(i + 1) % 2]
                for b in xrange(bottom[0].shape[0]):
                    x = top_diff[b]
                    y = data[b]
                    t1_, t2_ = t1, t2
                    if swap:
                        x, y = y, x
                        t1_, t2_ = t2_, t1_
                    linalg.dot(x, y,
                               transa=blas_trans(t1_), transb=blas_trans(t2_),
                               handle=h, out=diff[b])

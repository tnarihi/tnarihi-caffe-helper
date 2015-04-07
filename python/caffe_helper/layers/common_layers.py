import numpy as np
from pycuda.elementwise import ElementwiseKernel

import caffe
from caffe import Layer
import caffe.pycuda_util as pu

from vision_layers import DownSamplingLayer  # For backward compatibility


class ReshapeLayer(Layer):

    """Reshape

    You should specify indentical blobs for bottom and top
    """

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.shape_ = param['shape']
        for i, s in enumerate(self.shape_):
            if i != 0 and s < 0:
                raise ValueError(
                    "-1 is only allowed at 1st axis: %s" % str(self.shape_))
        if self.shape_[0] < 0:
            assert np.prod(self.shape_[1:]) == np.prod(bottom[0].shape[1:])
        else:
            assert np.prod(self.shape_) == np.prod(bottom[0].shape)
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        self.bottom_shape_ = bottom[0].shape
        if self.shape_[0] < 0:
            top[0].reshape(self.bottom_shape_[0], *self.shape_[1:])
        else:
            top[0].reshape(*self.shape_)

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].reshape(*self.bottom_shape_)


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


class ParameterLayer(Layer):

    """
    ParameterLayer is holding a parameter blob and feeds the data of the blob
    to the top blob directly. Note this always accumulates param grad, so it
    needs accum-grad branch.
    """

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.shape_ = param['shape']
        assert len(bottom) == 0
        assert len(top) == 1
        self.reshape(bottom, top)

        # Initialize parameter
        if len(self.blobs) > 0:
            assert self.blobs[0].shape == self.shape_
            return
        seed = param.get('seed', 313)
        rng = np.random.RandomState(seed)
        # filler must be a form of lambda shape, rng: <operation>
        filler = eval(param['filler'])
        self.blobs.append(caffe.Blob(self.shape_))
        self.blobs[0].data[...] = filler(top[0].data.shape, rng)

    def reshape(self, bottom, top):
        top[0].reshape(*self.shape_)

    def forward(self, bottom, top):
        top[0].data[...] = self.blobs[0].data

    def backward(self, top, propagate_down, bottom):
        self.blobs[0].diff[...] += top[0].diff


class ReductionLayer(Layer):

    """
    Parameters
    ----------

    :axis: Axis to be reduced
    :op: Operation of reduction. "mean" is only supported so far.
    """

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.axis_ = param['axis']
        self.op_ = param['op']
        if self.op_ not in ['mean']:
            raise ValueError("Unsupported op type: %s" % self.op_)
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        assert len(bottom) == 1
        assert len(top) == 1
        assert len(bottom[0].shape) >= self.axis_
        shape = list(bottom[0].shape)
        shape[self.axis_] = 1
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        if self.op_ == 'mean':
            top[0].data[...] = \
                bottom[0].data.mean(self.axis_).reshape(top[0].shape)
        else:
            raise ValueError("Unsupported op type: %s" % self.op_)

    def backward(self, top, propagate_down, bottom):
        if not propagate_down[0]:
            return
        if self.op_ == 'mean':
            bottom[0].diff[...] = top[0].diff / bottom[0].shape[self.axis_]
        else:
            raise ValueError("Unsupported op type: %s" % self.op_)


class SliceByArrayLayer(Layer):

    """
    Slicing 1st axis with an integer array
    """

    def setup(self, bottom, top):
        from scipy.io import loadmat
        param = eval(self.param_str_)
        self.indexes_ = loadmat(param['path_mat'])[param['key']].flatten()
        assert np.unique(self.indexes_).size == self.indexes_.size, \
            'Indexes must be unique each other.'
        self.axis_ = param.get('axis', 1)
        assert self.axis_ == 1, 'Now only axis=1 is supported.'
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        assert len(bottom) == 1
        assert len(top) == 1
        assert len(bottom[0].shape) > self.axis_
        shape = list(bottom[0].shape)
        shape[self.axis_] = self.indexes_.size
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data[:, self.indexes_, ...]

    def backward(self, top, propagate_down, bottom):
        if not propagate_down[0]:
            return
        bottom[0].diff[...] = 0
        bottom[0].diff[:, self.indexes_, ...] = top[0].diff


class BroadcastLayer(Layer):

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.axis_ = param['axis']
        self.num_ = param['num']
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        assert len(bottom) == 1
        assert len(top) == 1
        assert bottom[0].shape[self.axis_] == 1
        shape = list(bottom[0].shape)
        shape[self.axis_] = self.num_
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data

    def backward(self, top, propagate_down, bottom):
        if not propagate_down[0]:
            return
        bottom[0].diff[...] = top[0].diff.sum(self.axis_, keepdims=True)


class TileLayer(Layer):

    def setup(self, bottom, top):
        param = eval(self.param_str_)
        self.axis_ = param['axis']
        self.num_ = param['num']
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        assert len(bottom) == 1
        assert len(top) == 1
        shape = list(bottom[0].shape)
        shape[self.axis_] *= self.num_
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        reps = [1 for _ in bottom[0].shape]
        reps[self.axis_] = self.num_
        top[0].data[...] = np.tile(bottom[0].data, reps)

    def backward(self, top, propagate_down, bottom):
        if not propagate_down[0]:
            return
        shape = bottom[0].shape
        shape2 = shape[:self.axis_] + (self.num_,) + shape[self.axis_:]
        bottom[0].diff[...] = top[0].diff.reshape(shape2).sum(self.axis_)

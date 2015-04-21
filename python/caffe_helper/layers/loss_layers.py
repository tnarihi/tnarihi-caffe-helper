import numpy as np

import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel

import caffe
from caffe import Layer
import caffe.pycuda_util as pu

dtype = np.float32


class ScaleInvariantL2LossLayer(Layer):

    """Scale Invariant L2 Loss which is described in NYU-depth paper.
    You must specify loss_weight in LayerParameter. 
    """

    def setup(self, bottom, top):
        assert len(bottom) == 3
        assert len(top) == 1
        # parameter
        param = eval(self.param_str_)
        self.lambda_ = param['lambda']
        # Create CUDA function
        with pu.caffe_cuda_context():
            self.k_masked_diff_ = ElementwiseKernel(
                "float *diff, float *pred, float *label, float *mask",
                "diff[i] = (pred[i] - label[i]) * mask[i]", 'masked_diff')
            self.k_squared_ = ElementwiseKernel(
                "float *diff, float *diff2",
                "diff2[i] = diff[i] * diff[i]", 'squared')
            self.k_ensure_mask_sum_ = ElementwiseKernel(
                "float *mask_sum",
                "mask_sum[i] = max(mask_sum[i], 1.0f)", 'ensure_mask_sum')
            # This should be computed more faster by cublasSdot
            self.k_sum_ = ReductionKernel(
                dtype, neutral="0",
                reduce_expr="a+b", map_expr="d[i]",
                arguments="float *d")
            self.k_squred_sum_ = ReductionKernel(
                dtype, neutral="0",
                reduce_expr="a+b", map_expr="d[i] * d[i]",
                arguments="float *d")
            self.k_div_sum_ = ReductionKernel(
                dtype, neutral="0",
                reduce_expr="a+b",
                map_expr="d[i] / m[i]",
                arguments="float *d, float *m")
            self.k_div_squared_sum_ = ReductionKernel(
                dtype, neutral="0",
                reduce_expr="a+b",
                map_expr="d[i] * d[i] / (m[i] * m[i])",
                arguments="float *d, float *m")
            func_backward = SourceModule(
                """
#include <caffe/util/device_alternate.hpp>
__global__ void backward(float *pred, float *label, float *mask,
  float *diff_sum, float *mask_sum, int count, int stride, int sgn,
  float lambda, float *diff) {
  CUDA_KERNEL_LOOP(i, count) {
    diff[i] = mask[i] * 2.0f * sgn / mask_sum[i / stride]
        * ((pred[i] - label[i])
            - lambda / mask_sum[i / stride] * diff_sum[i / stride]);
  }
}
""", include_dirs=pu.caffe_include_dirs).get_function("backward")
            func_backward.prepare("PPPPPiiifP")

            def _func_backward(pred, label, mask, ds, ms, sgn, diff):
                bg = pu.block_and_grid(pred.size)
                count = pred.size
                stride = pred.size / pred.shape[0]
                func_backward.prepared_call(
                    bg['grid'], bg['block'],
                    pred.gpudata, label.gpudata, mask.gpudata, ds.gpudata,
                    ms.gpudata, count, stride, sgn, self.lambda_, diff.gpudata)
            self.k_backward_ = _func_backward
        self.batch_size_ = 0
        self.dim_ = 0
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        with pu.caffe_cuda_context():

            batch_size = bottom[0].shape[0]
            if self.batch_size_ != batch_size:
                self.batch_size_ = batch_size
                self.diff_sum_ = gpuarray.zeros((batch_size, 1), dtype)
                self.diff2_sum_ = gpuarray.zeros((batch_size, 1), dtype)
                self.mask_sum_ = gpuarray.zeros((batch_size, 1), dtype)
            dim = int(np.prod(bottom[0].shape[1:]))
            if self.dim_ != dim:
                self.dim_ = dim
                self.multipier_sum_ = gpuarray.zeros((dim, 1), dtype)
                self.multipier_sum_.fill(dtype(1.0))
        top[0].reshape()

    def forward(self, bottom, top):
        """

        """
        with pu.caffe_cuda_context():
            h = caffe.cublas_handle()
            batch_size = bottom[0].shape[0]
            dim = bottom[0].count / bottom[0].shape[0]
            pred = bottom[0].data_as_pycuda_gpuarray()
            label = bottom[1].data_as_pycuda_gpuarray()
            mask = bottom[2].data_as_pycuda_gpuarray()
            # Use bottom[0,1].diff as temporary buffer
            diff = bottom[0].diff_as_pycuda_gpuarray()
            diff2 = bottom[1].diff_as_pycuda_gpuarray()
            # Compute diff
            self.k_masked_diff_(diff, pred, label, mask)
            self.k_squared_(diff, diff2)
            import scikits.cuda.linalg as linalg
            # This needs scikits.cuda 0.5.0a3 or later
            # (sudo) pip install scikits.cuda=>0.5.0a3
            linalg.dot(diff.reshape(batch_size, dim), self.multipier_sum_,
                       handle=h, out=self.diff_sum_)
            linalg.dot(diff2.reshape(batch_size, dim), self.multipier_sum_,
                       handle=h, out=self.diff2_sum_)
            linalg.dot(mask.reshape(batch_size, dim), self.multipier_sum_,
                       handle=h, out=self.mask_sum_)
            self.k_ensure_mask_sum_(self.mask_sum_)
            term1 = self.k_div_sum_(self.diff2_sum_, self.mask_sum_)
            term2 = self.k_div_squared_sum_(self.diff_sum_, self.mask_sum_)
            top[0].data[...] = (term1.get() - self.lambda_ * term2.get()) \
                / batch_size

    def backward(self, top, propagate_down, bottom):
        """
        Compute @f$\frac{\partial {\cal L}}{\partial y_bi}=\frac{\partial {\cal L}}{\partial d_i} \frac{\partial d_i} {\partial y_bi}@f$.
        @f$\frac{\partial {\cal L}}{\partial d_i}=\frac{2}{n}d_i' \left(d_i - \frac{\lambda}{n}\sum_j d_j\right).
        """
        with pu.caffe_cuda_context():
            pred = bottom[0].data_as_pycuda_gpuarray()
            label = bottom[1].data_as_pycuda_gpuarray()
            mask = bottom[2].data_as_pycuda_gpuarray()
            for i in xrange(len(bottom) - 1):
                if propagate_down[i]:
                    diff = bottom[i].diff_as_pycuda_gpuarray()
                    sgn = 1 if i == 0 else - 1
                    self.k_backward_(
                        pred, label, mask, self.diff_sum_, self.mask_sum_, sgn,
                        diff)


class DSSIMLayer(Layer):

    def setup(self, bottom, top):
        # assert caffe.check_mode_gpu()
        # from theano.sandbox.cuda import use
        # use('gpu%d' % caffe.get_device())
        import theano as tn
        import theano.tensor as T
        from theano.tensor.signal.conv import conv2d
        assert len(bottom) >= 2
        assert len(bottom) <= 3
        assert len(top) == 1
        # parameter
        self.K_ = [0.01, 0.03]
        self.L_ = 1.0
        param = eval(self.param_str_)
        self.hsize_ = param.get('hsize', 11)
        self.sigma_ = param.get('sigma', 1.5)
        assert self.hsize_ % 2 == 1
        hsize = self.hsize_
        sigma = self.sigma_
        C1 = (self.K_[0] * self.L_) ** 2
        C2 = (self.K_[1] * self.L_) ** 2
        # Creating gaussian filter
        x = np.exp(-0.5 * ((np.arange(hsize) - int(hsize / 2)) ** 2) /
                   (sigma ** 2))
        filt = x.reshape(-1, 1) * x.reshape(1, -1)
        filt /= filt.sum()

        # Build a Theano function which computes SSIM and its gradients wrt two
        # images
        simg1_in = T.ftensor3()
        simg2_in = T.ftensor3()

        if len(bottom) > 2:
            smask = T.ftensor3()
            sk = T.sum(simg1_in * simg2_in * smask) \
                / T.sum(simg1_in * simg1_in * smask)
            simg1 = sk * simg1_in * smask
            simg2 = simg2_in * smask
        else:
            sk = T.sum(simg1_in * simg2_in) \
                / T.sum(simg1_in * simg1_in)
            simg1 = sk * simg1_in
            simg2 = simg2_in
        sfilt = tn.shared(filt.astype(np.float32))
        smu1 = conv2d(simg1, sfilt)
        smu2 = conv2d(simg2, sfilt)
        smu1_sq = smu1 * smu1
        smu2_sq = smu2 * smu2
        smu1_mu2 = smu1 * smu2
        ssig1_sq = conv2d(simg1 * simg1, sfilt) - smu1_sq
        ssig2_sq = conv2d(simg2 * simg2, sfilt) - smu2_sq
        ssig12 = conv2d(simg1 * simg2, sfilt) - smu1_mu2
        sssim = (
            ((2 * smu1_mu2 + C1) * (2 * ssig12 + C2))
            / ((smu1_sq + smu2_sq + C1) * (ssig1_sq + ssig2_sq + C2))
        ).mean()
        sdssim = (1 - sssim) / 2
        gimg1, gimg2 = tn.grad(sdssim, [simg1_in, simg2_in])
        if len(bottom) > 2:
            self.fdssim_with_grad = tn.function(
                [simg1_in, simg2_in, smask], [sdssim, gimg1, gimg2])
        else:
            self.fdssim_with_grad = tn.function(
                [simg1_in, simg2_in], [sdssim, gimg1, gimg2])

    def reshape(self, bottom, top):
        assert bottom[0].shape == bottom[1].shape
        assert len(bottom[0].shape) == 4
        top[0].reshape()

    def forward(self, bottom, top):
        dssim = np.float64(0.0)
        for i in xrange(bottom[0].shape[0]):
            if len(bottom) > 2:
                s, g1, g2 = self.fdssim_with_grad(
                    bottom[0].data[i], bottom[1].data[i], bottom[2].data[i])
            else:
                s, g1, g2 = self.fdssim_with_grad(
                    bottom[0].data[i], bottom[1].data[i])
            dssim += s
            bottom[0].diff[i] = g1 / bottom[0].shape[0]
            bottom[1].diff[i] = g2 / bottom[1].shape[0]
        top[0].data[...] = dssim / bottom[0].shape[0]

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] *= top[0].diff
        bottom[1].diff[...] *= top[0].diff

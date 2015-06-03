
theano_initialized = False

def init_theano():
	"""Initialize Theano for Caffe
	"""
	global theano_initialized
	if theano_initialized:
		return
	import caffe
	from theano.sandbox.cuda import use
	assert caffe.check_mode_gpu()
	use('gpu%d' % caffe.get_device())
	theano_initialized = True

def blob_to_CudaNdArray(b, diff=False):
    from theano.sandbox import cuda
    data_ptr = long(b.gpu_data_ptr)
    diff_ptr = long(b.gpu_diff_ptr)
    strides = [1]
    for i in b.shape[::-1][:-1]:
        strides.append(strides[-1]*i)
    strides = tuple(strides[::-1])
    return cuda.from_gpu_pointer(data_ptr, b.shape, strides, b), \
        cuda.from_gpu_pointer(diff_ptr, b.shape, strides, b)

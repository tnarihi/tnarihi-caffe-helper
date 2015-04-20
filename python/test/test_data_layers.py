import numpy as np

import caffe
from caffe.proto import caffe_pb2


def test_hdf5_layer(tmpdir):
    td = tmpdir.mkdir('test_hdf5_layer')
    path_h5 = td.join('h5layer.in.h5').ensure().strpath
    n_sample = 5
    names = ['data%02d' % i for i in xrange(n_sample)]
    blob_name = 'blob1'
    batch_size = 2
    blob_shape = (1, 2, 3)
    data = np.random.rand(*((n_sample,) + blob_shape))
    import h5py
    with h5py.File(path_h5, 'w') as hd:
        for i, name in enumerate(names):
            hd.create_group(name)
            hd[name][blob_name] = data[i]
    import csv
    lpath_source = td.join('source.txt').ensure()
    csv.writer(lpath_source.open('w')).writerows(map(lambda x: [x], names))
    t = caffe.Blob([])
    bottom = []
    top = [t]
    # Create Layer
    lp = caffe_pb2.LayerParameter()
    lp.type = "Python"
    lp.python_param.module = "caffe_helper.layers.data_layers"
    lp.python_param.layer = "HDF5Layer"
    lp.python_param.param_str = str(dict(
        batch_size=batch_size, source=lpath_source.strpath, path_h5=path_h5,
        column_id=0, blob_name='blob1'))
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    j = 0
    for i in xrange(3):
        layer.Reshape(bottom, top)
        layer.Forward(bottom, top)
        assert top[0].shape == (batch_size,) + blob_shape
        for blob in top[0].data:
            j %= n_sample
            assert np.all(blob == data[j].astype(np.float32))
            j += 1

    # Shuffle: Values are not checked here so far.
    lp.python_param.param_str = str(dict(
        batch_size=batch_size, source=lpath_source.strpath, path_h5=path_h5,
        column_id=0, blob_name='blob1', shuffle=True, random_seed=313))
    layer = caffe.create_layer(lp)
    layer.SetUp(bottom, top)
    for i in xrange(3):
        layer.Reshape(bottom, top)
        layer.Forward(bottom, top)
        assert top[0].shape == (batch_size,) + blob_shape

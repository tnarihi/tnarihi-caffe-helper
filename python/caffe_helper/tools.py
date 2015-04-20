import os
import numpy as np

from .proto_creator import *
from .obsolete import *


def save_output_hdf5(blobs, proto_path, model_path, path_out, path_names,
                     h5mode='a', name_column=0, gpu=0, phase=None):
    import csv
    import caffe
    from h5py import File as H5File
    if phase is None:
        phase = caffe.TEST
    try:
        os.makedirs(os.path.dirname(path_out))
    except:
        pass
    names = map(lambda x: x[name_column], csv.reader(open(path_names, 'r')))
    with H5File(path_out, h5mode) as h5d:
        if gpu < 0:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(gpu)
        net = caffe.Net(proto_path, model_path, phase)
        i = 0
        while True:
            ret = net.forward(blobs=blobs)
            for s in xrange(ret[blobs[0]].shape[0]):
                if len(names) == i:
                    return
                try:
                    h5d.create_group(names[i])
                except ValueError:
                    pass
                for b in blobs:
                    try:
                        h5d[names[i]][b] = ret[b][s].copy()
                    except (ValueError, RuntimeError):
                        del h5d[names[i]][b]
                        h5d[names[i]][b] = ret[b][s].copy()
                i += 1


def draw_net_from_prototxt(path_proto, rankdir='LR', **proto_kw):
    """"""
    if path_proto.endswith('.jinja2'):
        path_proto = convert_prototxt_template(path_proto, **proto_kw)
    from google.protobuf import text_format
    import caffe
    import caffe.draw
    from caffe.proto import caffe_pb2
    from PIL import Image
    from StringIO import StringIO
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(path_proto).read(), net)
    png = caffe.draw.draw_net(net, rankdir, ext='png')
    img = np.asarray(Image.open(StringIO(png)))
    return img

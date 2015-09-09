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


train_command_base = """
#! /usr/bin/env python
# BEGIN EMBEDDED
path_pkl = "{path_pkl}"
weights = "{weights}"
default_cbatch = {cbatch}
# END ENBEDDED

import cPickle
import subprocess
from optparse import OptionParser

usage = 'Usage: %prog [options] gpuid'
parser = OptionParser(usage=usage)
parser.add_option(
    '-b', '--batch', dest='cbatch', type='int', default=default_cbatch,
    help='Computational batch size',
    )

options, args = parser.parse_args()
if len(args) != 1:
    parser.error("GPU id must be specified.")
gpu = int(args[0])

with open(path_pkl, 'rb') as fd:
    pc, spc, prefix = cPickle.load(fd)
if options.cbatch is not None:
    cbatch_old = pc.kw['batch_size']
    pc.update(batch_size=options.cbatch)
    iter_size = spc.kw['accum_grad']
    lbatch = cbatch_old * iter_size
    assert lbatch % options.cbatch == 0
    spc.update(accum_grad=lbatch/options.cbatch)
net_proto = pc.create(prefix_proto=prefix)
subprocess.call('bash -c "' + spc.train_command(net_proto, weights, gpu=gpu) + '"', shell=True)
"""
def create_train_command(pc, spc, params, epochs, batch, prefix, num_train, weights):
    pc = pc.copy_and_update(params=params)
    cbatch = pc.kw['batch_size']
    spc = spc.copy_and_update(
        snapshot_prefix=pc.get_path_proto_base(prefix),
        accum_grad=batch/cbatch,
        max_iter=epochs * num_train / batch + 1,)
    path_pkl = spc.get_path_proto_base() + '.pkl'
    path_py = spc.get_path_proto_base() + '.train.py'
    with open(path_pkl, 'wb') as fd:
        s = cPickle.dump([pc, spc, prefix], fd)
    with open(path_py, 'w') as fd:
        print >> fd, train_command_base.format(path_pkl=path_pkl, weights=weights, cbatch=cbatch)
    print 'python', path_py

def create_test_command(
    blobs, pc, prefix, model_path, path_out, path_names,
    h5mode='a', name_column=0, gpu=0, phase=None,
    params=None
):
    if params is not None:
        pc = pc.copy_and_update(params=params)
    path_pkl = pc.get_path_proto_base(prefix) + '.pkl'
    path_py = pc.get_path_proto_base(prefix) + '.test.py'
    with open(path_pkl, 'wb') as fd:
        s = cPickle.dump([pc, spc, prefix], fd)
    with open(path_py, 'w') as fd:
        print >> fd, train_command_base.format(path_pkl=path_pkl, weights=weights, cbatch=cbatch)
    print 'python', path_py

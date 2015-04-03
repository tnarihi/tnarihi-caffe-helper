import os
import glob
import sys
import datetime
from subprocess import Popen
import hashlib
pj = os.path.join

import numpy as np

import caffe_helper as ch


def call_subprocess(*args, **kw):
    """"""
    try:
        p = Popen(*args, **kw)
        p.communicate()
        ret = p.returncode
        if ret != 0:
            raise RuntimeError(
                'Process "%s" exit with status %d' % (str(args[0]), ret))
    finally:
        """This doen't ensure that child processes is killed when parent
        crashes"""
        if 'p' in locals() and p.returncode is None:
            p.terminate()
            p.kill()
            print >> sys.stderr, 'Process killed: pid=%d command="%s"' % (
                p.pid, str(args[0]))


def train(solver_text, gpu=-1, pretrained=None, snapshot=None):
    """
    Parameters
    ----------
    solver_text: str
        Pah to protobuf file of solver settings.
    gpu: int
        Device id of GPU.
    pretrained: str
        CAFFE model file *.caffemodel.
    snapshot: str
        CAFFE solver state file *.solverstate.

    Retuns
    ------
    None
    """
    assert(not (pretrained is not None and snapshot is not None))
    tokens = [
        ch.caffe_bin, 'train',
        '-solver', solver_text,
    ]
    if gpu >= 0:
        tokens += ['-gpu', gpu]
    if pretrained is not None:
        tokens += ['-weights', pretrained]
    if snapshot is not None:
        tokens += ['-snapshot', snapshot]
    tokens = map(str, tokens)
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    logname = 'train.%s.%s' % (now, os.path.basename(solver_text))
    fo = pj(ch.dir_log, logname + '.o')
    fe = pj(ch.dir_log, logname + '.e')
    if ch.verbose_:
        print 'run "%s"' % ' '.join(tokens)
    with open(fo, 'w') as fod, open(fe, 'w') as fed:
        call_subprocess(tokens, stdout=fod, stderr=fed)


def convert_prototxt_template(path_template, path_proto=None, **proto_kw):
    """"""
    if path_proto is None:
        m = hashlib.md5(open(path_template, 'rb').read())
        if proto_kw:
            m.update(str(proto_kw))
        path_proto = pj(ch.dir_proto_out, m.hexdigest() + '.prototxt')
    with open(path_proto, 'w') as fd:
        tmpl = ch.j2env.from_string(open(path_template).read())
        print >> fd, tmpl.render(**proto_kw)
    return path_proto


def get_solver_prototxt(net,
                        snapshot_prefix, max_iter, test_iter=200,
                        test_interval=100, base_lr=0.001,
                        display=50, momentum=0.9,
                        weight_decay=1e-6, snapshot=100, debug_info=False,
                        accum_grad=1, share_blobs=True,
                        force_cpu_momentum=False, lr_policy="fixed", power=0.5):
    """
    """
    kw = locals().copy()
    path_solver = convert_prototxt_template(
        pj(ch.dir_template, 'solver.prototxt.jinja2'), **kw)
    return path_solver


def get_iter_from_path(pathi):
    return int(pathi.split('_')[-1].split('.')[0])


def get_models_and_states_from_prefix(snapshot_prefix):
    models = sorted(
        glob.glob(snapshot_prefix + "_iter_*.caffemodel"),
        key=CaffeHelper.get_iter_from_path)
    states = sorted(glob.glob(snapshot_prefix + "_iter_*.solverstate"),
                    key=CaffeHelper.get_iter_from_path)
    assert len(models) == len(states)
    iters = map(CaffeHelper.get_iter_from_path, models)
    return list(zip(iters, models, states))


def train_by_net(net,
                 snapshot_prefix, max_iter, test_iter=200,
                 test_interval=100, base_lr=0.001,
                 display=50, momentum=0.9,
                 weight_decay=1e-6, snapshot=100, debug_info=True,
                 gpu=-1, path_pretrained=None, path_snapshot=None,
                 dry_run=False):
    """
    """
    kw = locals().copy()
    del kw['gpu']
    del kw['path_pretrained']
    del kw['path_snapshot']
    del kw['dry_run']
    path_solver = get_solver_prototxt(**kw)
    if not dry_run:
        train(path_solver, gpu=gpu,
              pretrained=path_pretrained, snapshot=path_snapshot)
    return get_models_and_states_from_prefix(snapshot_prefix)


def get_output(blobs, proto_path, model_path,
               num_sample=173519, gpu=0):
    """"""
    import caffe
    if gpu < 0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
    caffe.set_phase_test()
    net = caffe.Net(proto_path, model_path)
    num_iter = -1
    batch_size = -1
    i = 0
    outputs = dict(zip(blobs, [[] for i in xrange(len(blobs))]))
    while True:
        ret = net.forward(blobs=blobs)
        if num_iter < 0:
            batch_size = ret[blobs[0]].shape[0]
            num_iter = int(np.ceil(num_sample * 1.0 / batch_size))
        i += 1
        if num_iter == i:
            for b in blobs:
                outputs[b] += [ret[b][:num_sample % batch_size].copy()]
                outputs[b] = np.concatenate(outputs[b], axis=0)
            break
        for b in blobs:
            outputs[b] += [ret[b].copy()]
    return outputs

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
        outputs = dict(zip(blobs, [[] for i in xrange(len(blobs))]))
        i = 0
        while True:
            ret = net.forward(blobs=blobs)
            for s in xrange(ret[blobs[0]].shape[0]):
                if len(names) == i:
                    return
                h5d.create_group(names[i])
                for b in blobs:
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


def get_net_from_prototxt(path_proto, path_model=None):
    """"""
    if path_proto.endswith('.jinja2'):
        path_proto = convert_prototxt_template(path_proto)
    import caffe
    if path_model is None:
        return caffe.Net(path_proto)
    return caffe.Net(path_proto, path_model)


def run_create_lmdb(image_txt, image_root, lmdb_out, shuffle=False,
                    height=None, width=None):
    """"""
    env = os.environ.copy()
    env['GLOB_logtostderr'] = '1'
    tokens = [
        pj(caffe_root_, 'build/tools', 'convert_imageset'),
        '%s/' % image_root,
        '%s' % image_txt,
        lmdb_out,
        '--encoded',
    ]
    if shuffle:
        tokens += ['--seed', '313', '--shuffle']
    if height is not None:
        tokens += ['--resize_height', height]
    if width is not None:
        tokens += ['--resize_width', width]
    tokens = map(str, tokens)
    print 'run:', ' '.join(tokens)
    call_subprocess(tokens, env=env)

import os
import glob
import sys
import datetime
from subprocess import Popen
import hashlib
pj = os.path.join

import numpy as np


def j2filter_slice_list(value, slices):
    try:
        return [value[i] for i in slices]
    except TypeError:
        return value[:slices]


def j2filter_bool2str(value):
    if value:
        return 'true'
    else:
        return 'false'


def j2filter_to_int(value):
    return int(value)

J2FILTERS = {
    'max': max,
    'min': min,
    'slice_list': j2filter_slice_list,
    'bool2str': j2filter_bool2str,
    'to_int': j2filter_to_int,
}


def call_subprocess(*args, **kw):
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
        if p.returncode is None:
            p.terminate()
            p.kill()
            print >> sys.stderr, 'Process killed: pid=%d command="%s"' % (
                p.pid, str(args[0]))


class CaffeHelper(object):

    def __init__(self, caffe_root, verbose=True, dir_log=None):
        """
        Parameters
        ----------
        caffe_root: str
            The root folder of CAFFE.
        """
        self.caffe_root_ = caffe_root
        self.verbose_ = verbose
        self.dir_log_ = dir_log
        self.setup()

    def setup(self):
        self.caffe_bin_ = pj(self.caffe_root_, 'build/tools/caffe')
        if self.dir_log_ is None:
            self.dir_log_ = pj(self.caffe_root_, 'tmp', 'log')
        if not os.path.isdir(self.dir_log_):
            os.makedirs(self.dir_log_)

        # Setting path to Caffe python
        sys.path.append(pj(self.caffe_root_, 'python'))

        # Setting up jinja2
        path_jinja2 = os.path.abspath(
            pj(os.path.dirname(__file__), '..', '..', 'model_templates')
        )
        from jinja2 import Environment, FileSystemLoader
        env = Environment()
        env.filters.update(J2FILTERS)
        env.loader = FileSystemLoader(path_jinja2)
        self.j2env_ = env
        self.dir_proto_out_ = pj(self.caffe_root_, 'tmp', 'prototxt')
        self.dir_template_ = path_jinja2
        if not os.path.isdir(self.dir_proto_out_):
            os.makedirs(self.dir_proto_out_)

    def train(self, solver_text, gpu=-1, pretrained=None, snapshot=None):
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
            self.caffe_bin_, 'train',
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
        fo = pj(self.dir_log_, logname + '.o')
        fe = pj(self.dir_log_, logname + '.e')
        if self.verbose_:
            print 'run "%s"' % ' '.join(tokens)
        with open(fo, 'w') as fod, open(fe, 'w') as fed:
            call_subprocess(tokens, stdout=fod, stderr=fed)

    def convert_prototxt_template(self, path_template, path_proto=None,
                                  **proto_kw):
        """
        """
        if path_proto is None:
            m = hashlib.md5(open(path_template, 'rb').read())
            if proto_kw:
                m.update(str(proto_kw))
            path_proto = pj(self.dir_proto_out_, m.hexdigest() + '.prototxt')
        with open(path_proto, 'w') as fd:
            tmpl = self.j2env_.from_string(open(path_template).read())
            print >> fd, tmpl.render(**proto_kw)
        return path_proto

    def get_solver_prototxt(self, net,
                            snapshot_prefix, max_iter, test_iter=200,
                            test_interval=100, base_lr=0.001,
                            display=50, momentum=0.9,
                            weight_decay=1e-6, snapshot=100, debug_info=True):
        """
        """
        kw = locals().copy()
        del kw['self']
        path_solver = self.convert_prototxt_template(
            pj(self.dir_template_, 'solver.prototxt.jinja2'), **kw)
        return path_solver

    @staticmethod
    def get_iter_from_path(pathi):
        return int(pathi.split('_')[-1].split('.')[0])

    @staticmethod
    def get_models_and_states_from_prefix(snapshot_prefix):
        models = sorted(
            glob.glob(snapshot_prefix + "_iter_*.caffemodel"),
            key=CaffeHelper.get_iter_from_path)
        states = sorted(glob.glob(snapshot_prefix + "_iter_*.solverstate"),
                        key=CaffeHelper.get_iter_from_path)
        assert len(models) == len(states)
        iters = map(CaffeHelper.get_iter_from_path, models)
        return list(zip(iters, models, states))

    def train_by_net(self, net,
                     snapshot_prefix, max_iter, test_iter=200,
                     test_interval=100, base_lr=0.001,
                     display=50, momentum=0.9,
                     weight_decay=1e-6, snapshot=100, debug_info=True,
                     gpu=-1, path_pretrained=None, path_snapshot=None,
                     dry_run=False):
        """
        """
        kw = locals().copy()
        del kw['self']
        del kw['gpu']
        del kw['path_pretrained']
        del kw['path_snapshot']
        del kw['dry_run']
        path_solver = self.get_solver_prototxt(**kw)
        if not dry_run:
            self.train(path_solver, gpu=gpu,
                       pretrained=path_pretrained, snapshot=path_snapshot)
        return self.get_models_and_states_from_prefix(snapshot_prefix)

    def get_output(self, blobs, proto_path, model_path,
                   num_sample=173519, gpu=0):
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

    def draw_net_from_prototxt(self, path_proto, rankdir='LR', **proto_kw):
        """
        """
        if path_proto.endswith('.jinja2'):
            path_proto = self.convert_prototxt_template(path_proto, **proto_kw)
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

    def get_net_from_prototxt(self, path_proto, path_model=None):
        if path_proto.endswith('.jinja2'):
            path_proto = self.convert_prototxt_template(path_proto)
        import caffe
        if path_model is None:
            return caffe.Net(path_proto)
        return caffe.Net(path_proto, path_model)

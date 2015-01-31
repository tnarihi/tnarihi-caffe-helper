import os
import sys
import datetime
from subprocess import call
pj = os.path.join

import numpy as np

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
        env.loader = FileSystemLoader(path_jinja2)
        self.j2env_ = env
        self.dir_proto_out_ = pj(self.caffe_root_, 'tmp', 'prototxt')
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
        now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        logname = 'train.%s.%s' % (now, os.path.basename(solver_text))
        fo = pj(self.dir_log_, logname + '.o')
        fe = pj(self.dir_log_, logname + '.e')
        if self.verbose_:
            print 'run "%s"' % ' '.join(tokens)
        call(tokens, stdout=open(fo, 'w'), stderr=open(fe, 'w'))
        if self.verbose_:
            print '####### stdout'
            call(['tail', fo])
            print '####### stderr'
            call(['tail', fe])

    def convert_prototxt_template(self, path_template, path_proto, **kw):
        with open(path_proto, 'w') as fd:
            tmpl = self.j2env_.from_string(open(path_template).read())
            print >> fd, tmpl.render(**kw)

    def draw_net_from_prototxt(self, path_proto, rankdir='LR'):
        if path_proto.endswith('.jinja2'):
            import tempfile
            with tempfile.NamedTemporaryFile(
                    'w', dir=self.dir_proto_out_, delete=False) as fd:
                path_proto_tmp = fd.name
            self.convert_prototxt_template(path_proto, path_proto_tmp)
            path_proto = path_proto_tmp
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



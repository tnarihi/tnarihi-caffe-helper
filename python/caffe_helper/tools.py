import os
import datetime
from subprocess import call
pj = os.path.join


class CaffeTools(object):

    def __init__(self, caffe_root, verbose=True, path_log=None):
        """
        Parameters
        ----------
        caffe_root: str
            The root folder of CAFFE.
        """
        self.caffe_root_ = caffe_root
        self.verbose_ = verbose
        self.path_log_ = path_log
        self.setup()

    def setup(self):
        self.caffe_bin_ = pj(self.caffe_root_, 'build/tools/caffe')
        if self.path_log_ is None:
            self.path_log_ = pj(self.caffe_root_, 'tmp', 'log')
        if not os.path.isdir(self.path_log_):
            os.makedirs(self.path_log_)

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
        fo = pj(self.path_log_, logname + '.o')
        fe = pj(self.path_log_, logname + '.e')
        if self.verbose_:
            print 'run "%s"' % ' '.join(tokens)
        call(tokens, stdout=open(fo, 'w'), stderr=open(fe, 'w'))
        if self.verbose_:
            print '####### stdout'
            call(['tail', fo])
            print '####### stderr'
            call(['tail', fe])

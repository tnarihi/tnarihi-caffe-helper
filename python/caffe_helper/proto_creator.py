import hashlib
from os.path import join
import caffe_helper as ch


class ProtoCreator(object):

    def __init__(self, proto_base, **kw):
        self._kw = kw.copy()
        self._proto_base = proto_base

    def create(self, path_proto=None):
        return convert_prototxt_template(
            self.proto_base, path_proto=path_proto, **self._kw)

    def copy(self):
        return self.__class__(proto_base=self.proto_base, **self._kw.copy())

    def update(self, **kw):
        self._kw.update(**kw)

    def copy_and_update(self, proto_base=None, **kw):
        cp = self.copy()
        if proto_base is not None:
            cp.proto_base = proto_base
        cp.update(**kw)
        return cp

    @property
    def proto_base(self):
        return self._proto_base

    @proto_base.setter
    def proto_base(self, value):
        self._proto_base = value

    @property
    def kw(self):
        return self._kw


class SolverProtoCreator(ProtoCreator):
    SOLVER_PROTO_BASE = join(ch.dir_template, 'solver.prototxt.jinja2')
    def __init__(
            self,
            snapshot_prefix, max_iter, test_iter=200,
            test_interval=100, base_lr=0.001,
            display=50, momentum=0.9,
            weight_decay=1e-6, snapshot=100, debug_info=False,
            accum_grad=1, share_blobs=True,
            force_cpu_momentum=False, lr_policy="fixed", power=0.5,
            proto_base=None, net=None):
        del proto_base
        del net
        kw = locals().copy()
        del kw['self']
        proto_base = SolverProtoCreator.SOLVER_PROTO_BASE
        super(SolverProtoCreator, self).__init__(proto_base, **kw)

    @ProtoCreator.proto_base.setter
    def proto_base(self, value):
        raise AttributeError("Assignment of `proto_base` is not allowed.")

    def create(self, net, path_proto=None):
        self._kw.update({'net': net})
        return super(SolverProtoCreator, self).create(path_proto)

    def model_path(self, iter_=None):
        if iter_ is None:
            iter_ = self._kw['max_iter'] + 1
        return self._kw['snapshot_prefix'] + '_iter_%d.caffemodel' % (iter_)

    def train_command(self, net_proto, weights=None, snapshot=None, gpu=0):
        com = ["%s/build/tools/caffe train" % ch.caffe_root,
               "-solver %s" % self.create(net_proto)]
        if weights is not None:
            com += ["-weights %s" % weights]
        if snapshot is not None:
            com += ["-snapshot %s" % snapshot]
        if gpu >= 0:
            com += ["-gpu %d" % gpu]
        return ' '.join(
            com + ['|& tee %s' % (self._kw['snapshot_prefix'] + '.log')])


def convert_prototxt_template(path_template, path_proto=None, **proto_kw):
    """"""
    if path_proto is None:
        m = hashlib.md5(open(path_template, 'rb').read())
        if proto_kw:
            m.update(str(proto_kw))
        path_proto = join(ch.dir_proto_out, m.hexdigest() + '.prototxt')
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
                        force_cpu_momentum=False, lr_policy="fixed", power=0.5,
                        path_proto=None):
    """
    """
    kw = locals().copy()
    path_solver = convert_prototxt_template(
        join(ch.dir_template, 'solver.prototxt.jinja2'), **kw)
    return path_solver

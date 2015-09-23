from __future__ import print_function
import glob
import re
import subprocess
import sys
from os.path import expandvars


class Tee(object):

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--weights')
    parser.add_argument('--solver')
    parser.add_argument('--gpu')
    parser.add_argument('--log')

    args = parser.parse_args()
    assert args.mode == 'train', 'only train supported'
    return args


def get_solver(args):
    from caffe.proto.caffe_pb2 import SolverParameter
    from google.protobuf.text_format import Merge

    solver = SolverParameter()
    Merge(open(args.solver, 'r').read(), solver)
    return solver


def get_iter_reached(args, solver):

    models = glob.glob(solver.snapshot_prefix + '_iter_*.solverstate')
    r = re.compile(r"^.+_iter_([0-9]+)\.solverstate$")
    iter_reached = 0
    for model in models:
        m = r.match(model)
        itr = int(m.groups()[0])
        iter_reached = max(itr, iter_reached)
    return iter_reached


def create_command(args, solver, iter_reached):

    cmd = [expandvars('$CAFFE_ROOT/build/tools/caffe'), args.mode]
    cmd += ['-solver', args.solver]
    if iter_reached == 0:
        if args.weights is not None:
            cmd += ['-weights', args.weights]
    else:
        cmd += ['-snapshot',
                solver.snapshot_prefix +
                '_iter_{}.solverstate'.format(iter_reached)]
    if args.gpu is not None:
        cmd += ['-gpu', args.gpu]
    return cmd


def main():
    args = parse_args()
    solver = get_solver(args)
    iter_reached = get_iter_reached(args, solver)

    if iter_reached >= solver.max_iter - 1:
        print('caffex.py DONE.')
        return

    cmd = create_command(args, solver, iter_reached)

    stdout = sys.stdout
    if args.log is not None:
        fd_log = open(args.log, 'a')
        stdout = Tee(fd_log, sys.stdout)
    print('run:', ' '.join(cmd))
    subprocess.call(cmd, stdout=stdout)

    return

main()

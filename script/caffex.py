from __future__ import print_function
import glob
import re
import subprocess
import sys
from os.path import expandvars
import signal
import time


class GracefulKiller:

    """Stack overflow : http://goo.gl/mdh4hP"""

    def __init__(self):
        print("killer init")
        self.kill_now = False
        # signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("Signal handler called")
        self.kill_now = True


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
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--log')
    parser.add_argument('--max_iter', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--warn_step_second', type=int, default=25)
    parser.add_argument('--time_limit', type=int)
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
        cmd += ['-gpu', str(args.gpu)]
    return cmd


def train(args, solver, iter_reached):

    import caffe

    if args.gpu is not None:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
    max_iter = solver.max_iter
    if args.max_iter is not None:
        max_iter = args.max_iter
    step_size = 1
    if args.step_size is not None:
        step_size = args.step_size

    print('init solver')
    s = caffe.SGDSolver(args.solver)

    # Resume from solverstate or weights
    if iter_reached == 0:
        if args.weights is not None:
            s.net.copy_from(args.weights)
    else:
        s.restore(str(solver.snapshot_prefix
                      + '_iter_{}.solverstate'.format(iter_reached)))

    # killer = GracefulKiller()  # Register signal handler

    stime_all = time.time()
    while s.iter < max_iter:
        print(s.iter)
        stime = time.time()
        s.step(step_size)
        time_elapsed = time.time() - stime
        if time_elapsed > 25:
            import warnings
            warnings.warn(
                "solver.step(step_size={}) takes {} sec. It might not be able "
                "to take a snapshot even when SIGINT/SIGTERM is "
                "received.".format(step_size, time_elapsed), RuntimeWarning)
        """
        if killer.kill_now:
            print("SIGTERM/SIGINT received: taking snapshot...")
            s.snapshot()
            print("Snapshot done. System killing process...")
            sys.exit(0)
        """
        if args.time_limit is not None and \
                time.time() - stime_all > args.time_limit:
            print("Reached to time limit: taking snapshot...")
            s.snapshot()
            sys.exit(0)
    else:
        print("Reached to max_iter: taking snapshot...")
        s.snapshot()
        print ("Training done.")


def main():
    args = parse_args()
    solver = get_solver(args)
    iter_reached = get_iter_reached(args, solver)

    if iter_reached >= solver.max_iter - 1:
        print('caffex.py DONE.')
        return

    train(args, solver, iter_reached)

    """
    cmd = create_command(args, solver, iter_reached)
    print('run:', ' '.join(cmd))
    if args.log is not None:
        subprocess.call(' '.join(cmd) + '|&tee -a {}'.format(args.log), shell=True)
    else:
        subprocess.call(' '.join(cmd), shell=True)
    """

main()

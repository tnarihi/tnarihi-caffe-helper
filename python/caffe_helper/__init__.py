def __setup():
    """Caffe helper set up function"""
    import os
    import sys
    from jinja2 import Environment, FileSystemLoader

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

    j2filters = {
        'max': max,
        'min': min,
        'slice_list': j2filter_slice_list,
        'bool2str': j2filter_bool2str,
        'to_int': j2filter_to_int,
    }
    pj = os.path.join
    caffe_root = os.getenv("CAFFE_ROOT")
    if caffe_root is None:
        raise ValueError(
            "Before calling this module, you should set `CAFFE_ROOT` as "
            "your environmental variable.")
    caffe_bin = pj(caffe_root, 'build/tools/caffe')
    dir_log = pj(caffe_root, 'tmp', 'log')
    if not os.path.isdir(dir_log):
        os.makedirs(dir_log)
    # Setting path to Caffe python
    sys.path.append(pj(caffe_root, 'python'))

    # Setting up jinja2
    path_jinja2 = os.path.abspath(
        pj(os.path.dirname(__file__), '..', '..', 'model_templates')
    )
    env = Environment(extensions=['jinja2.ext.do'])
    env.filters.update(j2filters)
    env.loader = FileSystemLoader(path_jinja2)
    dir_proto_out = pj(caffe_root, 'tmp', 'prototxt')
    dir_template = path_jinja2
    if not os.path.isdir(dir_proto_out):
        os.makedirs(dir_proto_out)
    return caffe_root, caffe_bin, dir_log, env, dir_proto_out, dir_template,


caffe_root, caffe_bin, dir_log, j2env, dir_proto_out, dir_template = __setup()
verbose = True


def set_dir_log(dir_log_):
    global dir_log
    dir_log = dir_log_


def set_verbose(verbose_):
    global verbose
    verbose = verbose_

import caffe_helper.tools
import caffe_helper.visualize

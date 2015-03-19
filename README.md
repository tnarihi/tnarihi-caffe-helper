# Caffe helper

This repositry provides helper tools and additional layers for BVLC/caffe.
Basically it is developed for my own use.

## Set up

### @tnarihi's Caffe fork

This helper tools requires to use @tnarihi's branch of Caffe. You can get it
by running:

```shell
git clone git://github.com:tnarihi/caffe.git
cd caffe
git checkout -b master-snapshost origin/master-snapshot
```

Once you get it, you can install it as you do to install the original Caffe.

### Other additional dependencies

Some Python layers of caffe requires additional dependencies.

* For CUDA
  * PyCuda >= 2014.1
  * Moca
  * Scikits.Cuda >= 0.5.0a3

* PyZMQ >= 14.5.0 (if you work with ipython)

* pytest (for testing)

### Setting paths

To use caffe and tnarihi-caffe-helper, you should set some environmental vars.
If you work on bash:

```shell
echo "export CAFFE_ROOT=<your caffe directory>" >> ~/.bashrc
echo "export PYTHONPATH=<your caffe directory>/python:<your caffe helper directory>/python" >> ~/.bashrc
```

### Test everythinig works
After installing them, run py.test at python folder.

```shell
cd <your caffe helper directory>/python
py.test
```

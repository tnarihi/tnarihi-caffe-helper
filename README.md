# Caffe helper

This repositry provides helper tools and additional layers for BVLC/caffe.
Basically it is developed for my own use, so documentation is quite poor and
codes are not so clean.

## Set up

### @tnarihi's Caffe fork

This helper tools require to use @tnarihi's branch of Caffe. You can get it
by running:

```shell
git clone git://github.com:tnarihi/caffe.git
cd caffe
git checkout -b master-snapshost origin/future
```

Once you get it, you will follow the installation of python dependencies in the
below section, then you can install it as you do to install the original Caffe.

### Python dependencies.

Some Python layers of caffe require additional dependencies.

[Here on Gist](https://gist.github.com/tnarihi/cf9154357500de8b051b) is a
script which installs the dependencies of this tools. This will install
Miniconda Python distribution in your home directory, and install the
dependencies. If you already have Anaconda/Miniconda, you might be able to skip
the python installation step in the script. I've confirmed this works on Ubuntu
12.04/14.04. I am installing OpenCV using conda but I recommend you to install
OpenCV from source because there would be some library conflicts. There might 
be some non-Python dependencies needed to be installed.

### Setting paths

To use caffe and tnarihi-caffe-helper, you should set some environmental vars.
If you work on bash:

```shell
echo "export CAFFE_ROOT=<your caffe directory>" >> ~/.bashrc
echo "export PYTHONPATH=<your caffe directory>/python:<your caffe helper directory>/python":$PYTHONPATH >> ~/.bashrc
```

### Test everythinig works
After installing them, run py.test at python folder.

```shell
cd <your caffe helper directory>/python
py.test
```

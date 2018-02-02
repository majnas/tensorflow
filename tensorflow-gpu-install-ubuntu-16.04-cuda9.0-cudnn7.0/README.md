# Base instruction is form https://github.com/williamFalcon
# This instruction is for installing CUDA 9.0 and cudnn 7.0 on ubuntu 16.04 for tensorflow-gpu framework


0- Update apt-get
sudo apt-get update

1- Install dependencies
sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev

2- Install nvidia drivers (for installing tensorflow from pre compiled binary file cuda 9.0 is better now)
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
dpkg -i ./cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
apt-get update

# Added this to make sure we don't drag down the newest version of cuda!
apt-get install cuda=9.0.176-1 -y

2a- Reboot ubuntu
sudo reboot

2b- Check nvidia driver install
nvidia-smi

# you should see a list of gpus printed
# if not, the previous steps failed.


3- Install cuda toolkit (MAKE SURE TO SELECT N TO INSTALL NVIDIA DRIVERS)

wget https://s3.amazonaws.com/personal-waf/cuda_9.0.176_384.81_linux.run   
sudo sh cuda_9.0.176_384.81_linux.run   # press and hold s to skip agreement   

# Do you accept the previously read EULA?
# accept

# Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 361.62?
# ************************* VERY KEY ****************************
# ******************** DON"T SAY Y ******************************
# n

# Install the CUDA 8.0 Toolkit?
# y

# Enter Toolkit Location
# press enter


# Do you want to install a symbolic link at /usr/local/cuda?
# y

# Install the CUDA 8.0 Samples?
# y

# Enter CUDA Samples Location
# press enter    

# now this prints: 
# Installing the CUDA Toolkit in /usr/local/cuda-8.0 …
# Installing the CUDA Samples in /home/liping …
# Copying samples to /home/liping/NVIDIA_CUDA-8.0_Samples now…
# Finished copying samples


4- Install cudnn
Download cudnn-9.0-linux-x64-v7.tgz from https://developer.nvidia.com/rdp/cudnn-download
# wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.tgz  
sudo tar -xzvf cudnn-9.0-linux-x64-v7.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

5- Add these lines to end of ~/.bashrc:
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH


6- Reload bashrc
source ~/.bashrc

# Check if the driver is installed currectly
nvcc -V
 

7- Install Anaconda 4.2.0 (Python 3.5)
wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
bash Anaconda3-4.2.0-Linux-x86_64.sh

# press s to skip terms
# Do you approve the license terms? [yes|no]
# yes

# Miniconda3 will now be installed into this location:
# accept the location

# Do you wish the installer to prepend the Miniconda3 install location
# to PATH in your /home/ghost/.bashrc ? [yes|no]
# yes 

8- Reload bashrc
source ~/.bashrc

9- Create conda env to install tensorflow
conda create -n tf15
# press y a few times

10- Activate env
source activate tf15

11- Install tensorflow with GPU support for python 3.5

pip install tensorflow-gpu

# If the above fails, try the part below
# download tensorflow_gpu-1.5.0-cp35-cp35m-manylinux1_x86_64 from https://pypi.python.org/pypi/tensorflow-gpu
# pip install --ignore-installed --upgrade tensorflow_gpu-1.5.0-cp35-cp35m-manylinux1_x86_64.whl

12- Test tf install
# start python shell
python

# run test script   
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# you will see list of GPUs using by tensorflow after executing tf.Session()

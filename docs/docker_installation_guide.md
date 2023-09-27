# How to setup docker container on your personal computer for the course

## What is container?

Containers are a lighter weight virtualization technology based on Linux namespaces. Unlike virtual machines, containers share the kernel and other services with the host. As a result, containers can startup very quickly and have negligible performance overhead, but they do not provide the full isolation of virtual machines.

Containers bundle the entire application user space environment into a single image. This way, the application environment is both portable and consistent, and agnostic to the underlying host system software configuration. Container images can be deployed widely, and even shared with others, with confidence that the results will be reproducible.

Containers make life simpler for developers, users, and system administrators. Developers can distribute software in a container to provide a consistent runtime environment and reduce support overhead. Container images from repositories such as NGC can help users start up quickly on any system and avoid the complexities of building from source. Containers can also help IT staff tame environment module complexity and support legacy workloads that are no longer compatible with host operating system.

## Install docker on host machine

There are many container runtimes, and the one we are about to use in this course is called **Docker**.

Refer to the official website: https://www.docker.com/

## How to setup docker container for this course

We have provided two different docker images for you. One for students with NVIDIA-card-equipped computer and the other forstudents without NVIDIA-card-equipped computer.

For both containers, you can use either `docker pull` from Docker Hub online (maybe slow)
or download the .tar file from the cluster login node, and use `docker load` to build image

**Directory of Images .tar file on the Cluster**
- With NVIDIA card (about 9GB):\
  10.26.200.21:/CSC4005-resources/nvhpc:21.7-devel-cuda11.4-centos7.tar\
  All six parallel languages supported


- Without NVIDIA card (about 670MB):\
  10.26.200.21:/CSC4005-resources/centos7_csc4005.tar\
  Only vectorization, MPI, Pthread, and OpenMP supported

**Notes:**
By default, docker stores all the images under C:\. Please make sure you have enough disk space in 'C:\' or refer to this instruction to try to store docker images in another disk.
[Instruction to change the docker default image location](#change-the-docker-default-image-location)

### If your laptop has an NVIDIA card and supports CUDA

This docker is able to run all the six parallel programming languages (AVX-512 instruction-set support depends on your CPU).

#### Prerequisite: check if you have an compatible NVIDIA driver installed

You need to install NVIDIA driver on your personal computer first, which supports CUDA 11.4.

Refer to the official driver releases and search for the one you need according to the NVIDIA card you have.
https://www.nvidia.com/Download/index.aspx?lang=en-us

```bash
nvidia-smi
Sun Sep 10 21:19:27 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 528.49       Driver Version: 528.49       CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   54C    P8     5W /  80W |     21MiB /  8192MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

Here, please check that the CUDA version supported by your NVIDIA driver is newer than CUDA-11.4.
Otherwise, you need to update your NVIDIA driver first if you want to use the nvhpc container.

**Installing Docker Image & Container**

```bash
# Pull docker image from nvidia
# It may be slow pulling images from Docker hub.
# As an alternative, we can download the .tar file from the cluster
#   Directory on the cluster: 10.26.200.21:/CSC4005-resources/nvhpc:21.7-devel-cuda11.4-centos7.tar
# After downloading, use docker load to build docker image from the .tar file
docker pull nvcr.io/nvidia/nvhpc:21.7-devel-cuda11.4-centos7
# or
docker load -i /path/to/nvcr.io/nvidia/nvhpc:21.7-devel-cuda11.4-centos7.tar

# Check the docker image pulled just now
docker images
# Terminal output
#    REPOSITORY             TAG                             IMAGE ID       CREATED       SIZE
#    nvcr.io/nvidia/nvhpc   21.7-devel-cuda11.4-centos7     2306fd2cb44f   2 years ago   9.62GB

# Build a docker container instance from the image
# Here, you need to set the <IMAGE ID> as the one displayed as the result of `docker images`, 2306fd2cb44f in this case
# Here, we bind port 22 of the docker container to the host machine as port 2222, so that we can SSH to the container through port 2222
docker run -it -d -p 2222:22 --gpus all --name csc4005 <IMAGE ID>
# If you want to mount a shared folder to the container, please add --volume flag
docker run -it -d -p 2222:22 --gpus all --name csc4005 --volume /path/host/directory:/path/container/directory <IMAGE ID>

# Check the docker container instance created just now
docker container ps
# Terminal output
#  CONTAINER ID   IMAGE          COMMAND                   CREATED             STATUS             PORTS                  NAMES
#  cf49d1025aff   2306fd2cb44f   "/usr/local/bin/entr…"   2 hours ago         Up 2 hours         0.0.0.0:2222->22/tcp   csc4005

# Execute the docker in bash terminal for interactive
# Here, you need to set the <CONTAINER ID> as the one displayed as the result of `docker container ps`, cf49d1025aff in this case
docker exec -it <CONTAINER ID> bash

# Now, you can interact with the docker container just like a normal linux machine.
# The followings are essential if your host's cuda driver version is high (e.g., latest 537.xx). In some older versions, they may be not necessary. You can check by yourself. 
echo "export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/bin:$PATH" >> ~/.bashrc
echo "export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin:$PATH" >> ~/.bashrc
echo "export CUDA_TOOLKIT_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4" >> ~/.bashrc
source ~/.bashrc

# [Optional] If you login the cluster with SSH, your $PATH may be incomplete. In this case, please execute the following commands
echo export 'LD_LIBRARY_PATH'=`echo $LD_LIBRARY_PATH`:'$LD_LIBRARY_PATH' >> ~/.bashrc
echo export 'PATH'=`echo $PATH`:'$PATH' >> ~/.bashrc
source ~/.bashrc

# Check if there are a long string output:
echo $PATH
echo $LD_LIBRARY_PATH

# We should see expected output for the following commands
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2021 NVIDIA Corporation    
# Built on Wed_Jun__2_19:15:15_PDT_2021
# Cuda compilation tools, release 11.4, V11.4.48
# Build cuda_11.4.r11.4/compiler.30033411_0     

pgc++ --version
# pgc++ (aka nvc++) 21.7-0 64-bit target on x86-64 Linux -tp haswell 
# PGI Compilers and Tools
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

nvc++ --version
# nvc++ 21.7-0 64-bit target on x86-64 Linux -tp haswell
# NVIDIA Compilers and Tools
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

# To compile OpenACC program in the container, you need to explicitly specify the CUDA version
# On the cluster, you don't need this '-gpu cuda11.4' flag
pgc++ -gpu cuda11.4 -acc -mp ./openacc_parallel.cpp -o openacc_parallel

# To run mpi program in the container, you need use the following command:
mpirun -n 4 --allow-run-as-root ./mpi_hello
```

### If your laptop does not have NVIDIA card equipped

This docker cannot compile & execute CUDA and OpenACC programs, but are fine with the other 4 programming languages.

```bash
# Pull a docker image of CentOS7 with basic MPI, Pthread, and OpenMP installed
# It may be slow pulling images from Docker hub.
# As an alternative, we can download the .tar file from the cluster
#   Directory on the cluster: 10.26.200.21:/CSC4005-resources/centos7_csc4005.tar
# After downloading from the cluster, use docker load to build the image
docker pull tonyyxliu/csc4005:centos7
# or
docker load -i /path/to/centos7_csc4005.tar

# Check the image you just pulled
docker images

# Build a docker container instance from the image
# Here, you need to set the <IMAGE ID> as the one displayed as the result of `docker images`, 2306fd2cb44f in this case
# Here, we bind port 22 of the docker container to the host machine as port 2222, so that we can SSH to the container through port 2222
docker run -it -d -p 2222:22 --name csc4005 <IMAGE ID>

# Check the docker container instance created just now
docker container ps

# Execute the docker in bash terminal for interactive
# Here, you need to set the <CONTAINER ID> as the one displayed as the result of `docker container ps`, cf49d1025aff in this case
docker exec -it <CONTAINER ID> bash
```

## How to upgrade gcc and cmake in docker container

The programs of our projects need `cmake3` and `gcc-7` for compilation and execution.

```bash
# Install cmake3 with yum
yum install cmake3 -y
cmake3 --version # output should be 3.17.5
# Note: use cmake3 to build the cmake project in your docker container

# Install gcc/g++-7 with yum
yum install -y centos-release-scl*
yum install -y devtoolset-7-gcc*
scl -l
echo "export PATH=/opt/rh/devtoolset-7/root/usr/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
gcc -v # output should be 7.3.1
```

## [Optional] How to connect to your docker container with SSH

This is just an alternative to write programs on your docker container.
You can also mount a shared folder to the docker container and write programs
on your host machine directly.

```bash
# First, enter your container with `docker exec -it <CONTAINER ID> bash`

# Install openssh-server
yum install -y openssh-server

# Start sshd service
/usr/sbin/sshd
# You may get the following error
# [root@ b3426410ff43 /]# /usr/sbin/sshd
# Could not load host key: /etc/ssh/ssh_host_rsa_key
# Could not load host key: /etc/ssh/ssh_host_ecdsa_key
# Could not load host key: /etc/ssh/ssh_host_ed25519_key

# Execute the following commands to generate keys for SSH
ssh-keygen -q -t rsa -b 2048 -f /etc/ssh/ssh_host_rsa_key -N ''
ssh-keygen -q -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -N ''
ssh-keygen -t dsa -f /etc/ssh/ssh_host_ed25519_key -N ''

# Reset password for root user in docker container
passwd root

# Restart sshd service
/usr/sbin/sshd

# Finally, use ssh -p 2222 root@localhost to connect to your docker container
```

## Change the docker default image location

### Method-1: Configure in Docker Desktop

Please refer to the official document and the GitHub issue raised by SummerSunnyNight for more details.

Official Document: https://docs.docker.com/desktop/settings/windows/#resources

GitHub Issue #14: https://github.com/tonyyxliu/CSC4005-2023Fall/issues/14

### Method-2: Configure in the Terminal [Windows Only]

This method was verified in Windows 10 Professional Edition (higher than or equal to version 19044) and the method needs the docker installation with WSL 2.

Follow the instruction below, we can see that the default path of docker (/var/lib/docker) is the same as shown in Linux.

```bash
docker info

--- output ---
...
Docker Root Dir: /var/lib/docker
...
```
This means that docker relies on WSL for file mapping, so we need to modify docker's file mapping path through wsl, which can be understood as file mounting.

The docker-desktop-data disk image in WSL 2 mode is usually located at the following location:
```text
C:\Users\<Your Name>\AppData\Local\Docker\wsl\data\ext4.vhdx
```
Follow the instructions below to relocate it to a different drive/directory and keep all existing Docker data.
- Exit Docker Desktop, then, open a command prompt (cmd):
```cmd
wsl --list -v
```
You should be able to see the states and make sure all states are stopped.

```cmd
NAME                  STATE        VERSION
docker-desktop        Stopped      2
docker-desktop-data   Stopped      2
```

- Export docker-desktop-data to a file (backup image and related files) and then import back to WSL2, and set the path you want, use the following command:
```cmd
wsl --export docker-desktop-data "D:\\docker-desktop-data.tar"
wsl --unregister docker-desktop-data
wsl --import docker-desktop-data "D:\\<You like>" "D:\\docker-desktop-data.tar" --version 2
```
Please note that the C:\\Users\\<Your Name>\\AppData\\Local\\Docker\\wsl\\data\\ext4.vhdx file will be automatically deleted.

Now start Docker Desktop and it will work normally.
Don't forget to delete the D:\\docker-desktop-data.tar file last.


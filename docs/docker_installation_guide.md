# How to setup docker container on your personal computer for the course

## What is container?

Containers are a lighter weight virtualization technology based on Linux namespaces. Unlike virtual machines, containers share the kernel and other services with the host. As a result, containers can startup very quickly and have negligible performance overhead, but they do not provide the full isolation of virtual machines.

Containers bundle the entire application user space environment into a single image. This way, the application environment is both portable and consistent, and agnostic to the underlying host system software configuration. Container images can be deployed widely, and even shared with others, with confidence that the results will be reproducible.

Containers make life simpler for developers, users, and system administrators. Developers can distribute software in a container to provide a consistent runtime environment and reduce support overhead. Container images from repositories such as NGC can help users start up quickly on any system and avoid the complexities of building from source. Containers can also help IT staff tame environment module complexity and support legacy workloads that are no longer compatible with host operating system.

## Install docker on host machine

There are many container runtimes, and the one we are about to use in this course is called **Docker**.

Refer to the official website: https://www.docker.com/

## How to setup docker container for this course

### If your laptop has an NVIDIA card and supports CUDA

#### Prerequisite

You need to install NVIDIA driver on your personal computer first, which supports CUDA 11.4.

Refer to the official driver releases and search for the one you need according to the NVIDIA card you have.
https://www.nvidia.com/Download/index.aspx?lang=en-us


**How to check if there is an NVIDIA driver installed on my computer?**

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

# Check the docker container instance created just now
docker container ps
# Terminal output
#  CONTAINER ID   IMAGE          COMMAND                   CREATED             STATUS             PORTS                  NAMES
#  cf49d1025aff   2306fd2cb44f   "/usr/local/bin/entr…"   2 hours ago         Up 2 hours         0.0.0.0:2222->22/tcp   csc4005

# Execute the docker in bash terminal for interactive
# Here, you need to set the <CONTAINER ID> as the one displayed as the result of `docker container ps`, cf49d1025aff in this case
docker exec -it <CONTAINER ID> bash

# Now, you can interact with the docker container just like a normal linux machine.
# You need to do some manipulations on the system path for your nvcc and pgc++ to work as expected
echo 'export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

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
```

### If your laptop does not have NVIDIA card equipped
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

## How to connect to your docker container with SSH

```bash
# First, enter your container with `docker exec -it <CONTAINER ID> bash`

# Install openssh-server
yum install -y openssh-server

# Start sshd service
/usr/sbin/sshd -D
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
/usr/sbin/sshd -D

# Finally, use ssh -p 2222 root@localhost to connect to your docker container
```

# Deep Learning Institute (DLI) Teaching Kit Lab/solution Repository

Welcome to the DLI Teaching Kit Lab/solution repository. The kit and associated labs are produced jointly by NVIDIA and New York University (NYU).  All material is available under the [Creative Commons Attribution-NonCommercial License](http://creativecommons.org/licenses/by-nc/4.0/).

The labs are designed to be open-ended, multidisciplinary, one- to three-week programming and written assignments for students. Each lab contains a description, sample code, sample solutions and suggestions on how instructors can evaluate and have students submit solutions.

*Note that we are currently in the 2nd release of this Teaching Kit. If you have never pulled any labs from this repo and/or are new to the DLI Teaching Kit, please pull from the "Rel2" branch. The new 2nd release labs are 2B, 3, and 4B. Labs 2 and 4 have two different versions (“A” and “B”) because they contain some overlapping concepts but use different problem/data sets. Labs 2B and 4B utilize the PyTorch frameworks. Lab 3 covers Generative Adversarial Networks. Labs 2A and 4A are the same as 1st release Labs 2 and 4. Please pull the 2nd release branch to a new location locally if you have pulled from the previous release.*

## System Requirements

#### NVIDIA CUDA/GPUs

Thanks to the rapid development of NVIDIA GPUs, training deep neural networks is more efficient than ever in terms of both time and resource cost. Training neural networks on [NVIDIA CUDA-enabled GPUs](https://developer.nvidia.com/cuda-gpus) is a practical necessity for the Teaching Kit labs, including both convolutional networks (Lab1 and Lab2) and recurrent networks (Lab4).

**Don't have access to GPUs? The DLI Teaching Kit comes with codes worth up to $125 of Amazon Web Services (AWS) GPU compute credit for each student in your course, as well as $200 for yourself as the instructor, to provide a GPU compute platform** to work on the open-ended labs. To request a code for yourself and your students, please send an email to [NVDLI@nvidia.com](mailto: NVDLI@nvidia) with the subject line “DLI Teaching Kit AWS Access”. An email will follow with your code and instructions for giving access to your students. *Instructions on setting up your AWS environment can be found below.*

The use of GPUs for the Teaching Kit labs requires a CUDA supported operating system, C compiler, and a recent CUDA Toolkit. Basic instructions on how to download and install the CUDA Toolkit can be found below in the Environment setup. More details can be found in the NVIDIA documentation on how [download](https://developer.nvidia.com/cuda-downloads)and [install](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) the CUDA Toolkit. Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and [OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

CUDA and the associated libraries should be installed prior to any deep learning specific tools.

#### Environment setup

*Note these steps have only been tested on GPU equipped nodes.*

      Step 1: If using AWS or another cloud resource, create an computing instance.

      Step 2: Launch a terminal
    
      Step 3: Download and install CUDA Toolkit
        wget http://us.download.nvidia.com/tesla/375.51/nvidia-driver-local-repoubuntu1604_375.51-1_amd64.deb
        sudo dpkg -i nvidia-driver-local-repo-ubuntu1604_375.51-1_amd64.deb
        sudo apt-get update
        sudo apt-get -y install cuda-drivers

      Step 4: Reboot the node

      Step 5: Install Anaconda, see: https://docs.continuum.io/anaconda/install
        wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
        bash ~/Downloads/Anaconda2-4.3.1-Linux-x86_64.sh

      Step 6: Install PyTorch and/or Torch as per instructions/links below

#### PyTorch and Torch computing frameworks
    
The DLI Teaching Kit labs and example solutions are based on the [PyTorch](http://pytorch.org/) and [Torch](http://torch.ch) computing frameworks. Please refer to [Getting started with Torch](http://torch.ch/docs/getting-started.html) for instruction on Torch installation, examples and documentation.

For Windows users, please refer to [Running Torch on Windows](https://github.com/torch/torch7/wiki/Windows#using-a-virtual-machine). At the time of writing this, PyTorch does not run on Windows, but there's an ongoing thread [here](https://github.com/pytorch/pytorch/issues/494).

#### cuDNN

The NVIDIA CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.

PyTorch integrates cuDNN automatically. To install cuDNN and use cuDNN with Torch, please follow the README on the [cuDNN Torch bindings](https://github.com/soumith/cudnn.torch) project.

## About the Labs/solutions

#### Recommended prerequisite DLI Teaching Kit lectures for each lab
* Lab1: Module 1 - Introduction to Machine Learning through Module 2 - Introduction to Deep Learning
* Lab2: Module 1 through Module 3 - Convolutional Neural Networks
* Lab3: Module 1 through Module 5 - Optimization Techniques
* Lab4: Module 1 through Module 6 - Learning with Memory

#### Lab documents
`documents` in each lab directory contains the same lab description and sample solution write-up `.pdf` documents as in the DLI Teaching Kit `.zip` package.

#### Baseline sample code
`sample_code` in each each lab directory contains the baseline training model solution (as well as instructions to run) described in the lab descriptions. These baseline models render a baseline score for the given data set that students are suggested to outperform. The `sample_code` is designed to be given to students when the lab is assigned.

#### Lab solutions
`solution_code` in each lab directory contains an example implementation of approaches that improve the model performance. These solutions were developed by real students who took the Deep Learning curriculum course at NYU. Some solutions may require additional, undocumented instructions to properly execute. 
Unlike the `sample_code`, some solution samples are not designed to run "out-of-box", but should still provide useful examples of solutions using a variety of techniques for both instructors and students to learn from.
However, the software structure remains the same as `sample_code` and uses the same execution script in the `sample_code` `Readme`s to run.
Note that for each lab, the sample solution code corresponds to only the 1st "lab*n*_*labName*_solution1.pdf" solution write-up. These solution write-ups are found in both the Teaching Kit `.zip` package and the `documents` folder in each lab directory in this repository.

#### Cloning and accessing the labs/solutions

To clone the Labs/solutions on your machine and, for example, access Lab1:
```
    git clone git@bitbucket.org:junbo_jake_zhao/deeplearningkit.git
    cd Lab1
```
#### In-class competition    
Some programming labs include optimizing a neural network training model and suggest students submit solutions to Kaggle using [Kaggle In Class](https://inclass.kaggle.com/) to compare inference accuracy against each other and against the baseline model score from the `sample_code`. Such a competition can encourage students to study the latest public research papers and technical reports to improve their model accuracy on an open-ended problem. Grading model accuracy could simply be based on whether they outperform the baseline, or perhaps based on class rank.

Please read the Kaggle In Class [FAQ](https://www.kaggle.com/about/inclass/faqs) for more information on how to set up your course using Kaggle. Using Kaggle is **not** a requirement to make use of the labs. For example, here is one way to evaluate lab solutions without Kaggle:

- Instructor creates (but does not release) a testing data set with the corresponding groundtruth prediction label file
- Students/teams develop models and compare model inference accuracy on a validation subset from a given training set (i.e. MNIST)
- Students/teams create a `result.lua` file that takes in their model file and the data set, and returns a model prediction in `.csv` format (see details in lab documents)
- Students/teams submit both their most accurate model and `result.lua` scripts
- Instructor executes the `result.lua` for each student/team's submitted model on the unreleased testing data set
- Compare the model prediction and groudtruth label on the testing set, and obtain the accuracy
- Use the testing accuracy to evaluate/compare students'/teams' model performance

## NVIDIA DLI Online Courses and Certification

The NVIDIA Deep Learning Institute (DLI) Teaching Kit includes access to free online DLI courses – **a value of up to $90 per person per course**. DLI training reinforces deep learning concepts presented in the Teaching Kits and teaches students how to apply those concepts to end-to-end projects. Through built-in assessments, students can earn certifications that prove subject matter competency and can be leveraged for professional career growth. Each course presents a self-paced learning environment with access to a GPU-accelerated workstation in the cloud. All you need is a web browser and Internet connection to get started.

The recommended DLI course (with certification) for students learning through the DLI Teaching Kit is **[Fundamentals of Deep Learning for Computer Vision](https://courses.nvidia.com/courses/course-v1:DLI+C-FX-01+V2/about)**.

`Syllabus.pdf` suggests students take this full-day course upon near-completion of your university semester course. It also suggests shorter courses that can be used as labs throughout your university course.

*To enable these or any other courses for your students, please send an email to NVDLI@nvidia.com with subject line “DLI Teaching Kit Online Course Access”. You will then receive information about how to give free access to your students.*

Detailed descriptions of all available DLI courses can be found at [www.nvidia.com/dli](https://www.nvidia.com/dli).

## About the NVIDIA Deep Learning Institute (DLI)
The NVIDIA Deep Learning Institute (DLI) offers hands-on training for developers, datascientists, and researchers looking to solve challenging problems with deep learning and accelerated computing. Through built-in assessments, students can earn certifications thatprove subject matter competency and can be leveraged for professional career growth.

#### Attend Instructor-led Training
In addition to online, self-paced courses, DLI offers all fundamentals and industry-specific courses as in-person workshops led by DLI-certified instructors. View upcoming workshops near you at [www.nvidia.com/dli](https://www.nvidia.com/dli).

Want to schedule training for your school? Request an onsite workshop at your university at [www.nvidia.com/requestdli](https://www.nvidia.com/requestdli).

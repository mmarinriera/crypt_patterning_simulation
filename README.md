# Agent based simulation of in-vitro intestinal crypt patterning

Powered by [ya||a](https://github.com/germannp/yalla)!

## Pre-requisites

- Linux OS.

- A CUDA-capable GPU.

- CUDA.

- The ya||a source code repository.



## Installation

- Clone the ya||a repository.

- Clone this repository in the same folder where the ya||a repository is, e.g.

```bash
some_folder/
├── crypt_patterning_simulation
└── yalla
```

- Compile the code by running the following command (ou may need to replace `sm_86` to a different value, depending on your specific GPU model),

```bash
nvcc -std=c++14 -arch=sm_86 crypt_patterning.cu
```

## Running the simulation

Run the simulation with sample parameter values with the following command.

```bash
./a.out 0 1.67 10 1.56 0.098 0.09 2.55 9.24 0
```

Simulation output files will be saved in the `output` folder inside the repository folder.

To visualize the simulation output, you need a 3D visualization software that supports VTK (e.g. [ParaView](https://www.paraview.org/download/)).

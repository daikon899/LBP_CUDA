![last commit](https://img.shields.io/github/last-commit/sim-pez/PRNU)
![](https://img.shields.io/badge/Programming_Language-c++-blue.svg)

# Intro 

This repository contains an implementation of [Local Binary Pattern algorithm](https://en.wikipedia.org/wiki/Local_binary_patterns) algorithm using GPU acceleration in CUDA C. This project is intended to compare speed performances wrt to CPU version ([sequential](https://github.com/MarcoSolarino/LBPSequential/tree/master) and [OpenMP](https://github.com/sim-pez/LBP_OpenMP)) of this algorithm. 


# Usage

- Place an image in .jpg format in ```input/``` folder
- Run the program specifying the image name like
```
LBPSequential img.jpg
```
- At the end of the run an histogram will be generated in ```output/```

# Tests

You should see something 



# Other versions

-[squential C++ version](https://github.com/MarcoSolarino/LBPSequential/tree/master) and 
-[OpenMP version](https://github.com/daikon899/LBP_OpenMP)


# Acknowledgments
Parallel Computing - Computer Engineering Master Degree @[University of Florence](https://www.unifi.it/changelang-eng.html)

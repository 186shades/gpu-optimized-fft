# High-Performance Implementation of the Fast Fourier Transform (FFT)

This repository contains the code and documentation for the **High-Performance Implementation of the Fast Fourier Transform Using the Cooley–Tukey Algorithm**. The project explores iterative and recursive implementations of the FFT, GPU-accelerated solutions, and various optimization techniques using CUDA.

## Overview
The primary objective of this project is to demonstrate the high-performance implementation of the FFT using:
1. **CPU-based Cooley-Tukey Algorithm** (Recursive and Iterative)
2. **Naive GPU Implementation**
3. **Optimized GPU Implementations**:
   - Shared memory and streams
   - Zero-padding to mitigate bank conflicts
   - Synchronization overhead reduction

This project provides a comparison of execution time between CPU and GPU implementations, showcasing how GPU optimizations drastically reduce computational overhead.

## Directory Structure
```
-- Build
-- Cmake
-- Extern
-- Src
|  +-- fftimpl
    |  +-- fftimpl.cu
```

## Prerequisites

- CUDA Enabled Machine
- Windows 10 x64
- CMake
- Visual Studio 16 2019

## Project Setup

a. Build project with CMake

/path/to/fft -> Configure -> Generate -> Open Project

b. Build Solution in Visual Studio

Release -> Build -> Build Solution

## Results
![](https://github.com/186shades/gpu-optimized-fft/blob/main/results/final_chart.png)

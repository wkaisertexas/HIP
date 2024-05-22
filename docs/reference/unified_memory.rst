.. meta::
  :description: This chapter describes introduces Unified Memory (UM) and shows
                how to use it in AMD HIP.
  :keywords: AMD, ROCm, HIP, CUDA, unified memory, unified, memory, UM, APU

*******************************************************************************
Unified Memory
*******************************************************************************

Concept
=======

In a conventional architectures, both CPUs a GPUs have a dedicated memory,
the RAM and VRAM, respectively. Inherent in this architectural design is the
need for continuous memory copying to allow the processors to access the
appropriate data. This reduces the maximum available memory capacity and
bandwidth. One way to avoid these effects is to use Heterogeneous System
Architectures (HSA) such as Unified Memory.

The Unified Memory (UM) can be understood as a single memory address space that
is accessible from any processor within a system. This setup enables
applications to allocate data that can be read or written by code running on
either CPUs or GPUs. The Unified memory model is shown in the figure below.

.. figure:: ../data/unified_memory/um.svg

AMD Accelerated Processing Unit (APU) is a typical example of a Unified Memory
Architecture. On a single die, a central processing unit (CPU) is combined with
an integrated graphics processing unit (iGPU) and both have the access for a
high bandwidth memory module, named as Unified Memory. The CPU enables
high-performance low latency operations, while the GPU is optimized for
high-throughput (data processed by unit time).

List of managed memory functions
================================

.. doxygengroup:: MemoryM
   :content-only:

Example for Unified Memory Management
=====================================

The following HIP program with unified memory management shows the addition of
two integers. In the other tab we can compare it to explicit memory management.

.. tab-set::

    .. tab-item:: Unified Memory Management

        .. code:: cpp

            #include <hip/hip_runtime.h>
            #include <iostream>

            // Addition of two values.
            __global__ void add(int *a, int *b, int *c) {
                *c = *a + *b;
            }

            int main() {
                int *a, *b, *c;

                // Allocate device copies of a, b and c.
                hipMallocManaged(&a, sizeof(*a));
                hipMallocManaged(&b, sizeof(*b));
                hipMallocManaged(&c, sizeof(*c));

                // Setup input values.
                *a = 1;
                *b = 2;

                // Launch add() kernel on GPU.
                hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, a, b, c);

                // Wait for GPU to finish before accessing on host.
                hipDeviceSynchronize();

                // Prints the result.
                std::cout << *a << " + " << *b << " = " << *c << std::endl;

                // Cleanup allocated memory.
                hipFree(a);
                hipFree(b);
                hipFree(c);

                return 0;
            }


    .. tab-item:: Explicit Memory Management

        .. code:: cpp

            #include <hip/hip_runtime.h>
            #include <iostream>

            // Addition of two values.
            __global__ void add(int *a, int *b, int *c) {
                *c = *a + *b;
            }

            int main() {
                int a, b, c;
                int *d_a, *d_b, *d_c;

                // Setup input values.
                a = 1;
                b = 2;

                // Allocate device copies of a, b and c
                hipMalloc(&d_a, sizeof(*d_a));
                hipMalloc(&d_b, sizeof(*d_b));
                hipMalloc(&d_c, sizeof(*d_c));

                // Copy input values to device.
                hipMemcpy(d_a, &a, sizeof(*d_a), hipMemcpyHostToDevice);
                hipMemcpy(d_b, &b, sizeof(*d_b), hipMemcpyHostToDevice);

                // Launch add() kernel on GPU.
                hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, d_a, d_b, d_c);

                // Copy the result back to the host.
                hipMemcpy(&c, d_c, sizeof(*d_c), hipMemcpyDeviceToHost);

                // Cleanup allocated memory.
                hipFree(d_a);
                hipFree(d_b);
                hipFree(d_c);

                // Prints the result.
                std::cout << a << " + " << b << " = " << c << std::endl;

                return 0;
            }


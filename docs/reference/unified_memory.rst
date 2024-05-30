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
high bandwidth memory (HBM) module, named as Unified Memory. The CPU enables
high-performance low latency operations, while the GPU is optimized for
high-throughput (data processed by unit time).

How-to use?
===========

Unified Memory Management (UMM) is a feature that can simplify the complexities
of memory management in GPU computing. It is particularly useful in
heterogeneous computing environments with heavy memory usage with both a CPU
and a GPU which would require large memory transfers. Here are some areas where
UMM can be beneficial:

- **Simplification of Memory Management**:
UMM can help to simplify the complexities of memory management. This can make
it easier for developers to write code without having to worry about the
details of memory allocation and deallocation.

- **Data Migration**:
UMM allows for efficient data migration between the host (CPU) and the device
(GPU). This can be particularly useful for applications that need to move data
back and forth between the device and host.

- **Improved Programming Productivity**:
As a positive side effect, the use of UMM can reduce the lines of code,
thereby improving programming productivity.

In HIP, pinned memory allocations are coherent by default. Pinned memory is
host memory that is mapped into the address space of all GPUs, meaning that the
pointer can be used on both host and device. Using pinned memory instead of
pageable memory on the host can lead an improvement in bandwidth.

While UMM can provide numerous benefits, it is also important
to be aware of the potential performance overhead associated with UMM.
Therefore, it is recommended to thoroughly test and profile your code to
ensure it is indeed the most suitable choice for your specific use case.

System Requirements
===================
Unified memory is supported on Linux by all modern AMD GPUs from the Vega
series onwards. Unified memory management can be achieved with managed memory
allocation and, for the latest GPUs, with a system allocator.

The table below lists the supported allocators. The allocators are described in
the next chapter.

.. csv-table::
    :widths: 50, 10, 10, 10
    :header: "GPU", "hipMallocManaged", "managed", "malloc"

        "MI200, MI 300 Series", "✅" , "✅" , "✅:sup:`1`"
        "MI100", "✅" , "✅" , "❌"
        "RDNA (Navi) Series", "✅" , "✅" , "❌"
        "GCN5 (Vega) Series", "✅" , "✅" , "❌"

✅: **Supported**

❌: **Unsupported**

:sup:`1` Works only with ``XNACK=1``. First GPU access causes recoverable page-fault.

Unified Memory Programming Models
=================================

- **HIP Managed Memory Allocation API**:
The ``hipMallocManaged()`` is a dynamic memory allocator available at all GPUs
with unified memory support.

- **HIP Managed Variables**:
The ``__managed__`` declaration specifier, which serves as its counterpart, is
supported across all modern AMD cards and can be utilized for static
allocation.

- **System Allocation API**:
Starting with the MI300 series, it is also possible to reserve unified memory
via the ``malloc()`` system allocator.

If it is wondered whether the GPU and the environment are capable of supporting
unified memory management, the ``hipDeviceAttributeConcurrentManagedAccess``
device attribute can answer it:

.. code:: cpp

    #include <hip/hip_runtime.h>
    #include <iostream>

    int main() {
        int d;
        hipGetDevice(&d);

        int is_cma = 0;
        hipDeviceGetAttribute(&is_cma, hipDeviceAttributeConcurrentManagedAccess, d);
        std::cout << "HIP Managed Memory: " << (is_cma == 1 ? "is" : "NOT") << " supported" << std::endl;
        return 0;
    }

Example for Unified Memory Management
-------------------------------------

The following example shows how to use unified memory management with
``hipMallocManaged()``, function, with ``__managed__`` attribute for static
allocation and standard  ``malloc()`` allocation. The Explicit Memory
Management is presented for comparison.

.. tab-set::

    .. tab-item:: hipMallocManaged()

        .. code:: cpp

            #include <hip/hip_runtime.h>
            #include <iostream>

            // Addition of two values.
            __global__ void add(int *a, int *b, int *c) {
                *c = *a + *b;
            }

            int main() {
                int *a, *b, *c;

                // Allocate memory for a, b and c that is accessible to both device and host codes.
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


    .. tab-item:: __managed__

        .. code:: cpp

            #include <hip/hip_runtime.h>
            #include <iostream>

            // Addition of two values.
            __global__ void add(int *a, int *b, int *c) {
                *c = *a + *b;
            }

            // Declare a, b and c as static variables.
            __managed__ int a, b, c;

            int main() {
                // Setup input values.
                a = 1;
                b = 2;

                // Launch add() kernel on GPU.
                hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, &a, &b, &c);

                // Wait for GPU to finish before accessing on host.
                hipDeviceSynchronize();

                // Prints the result.
                std::cout << a << " + " << b << " = " << c << std::endl;

                return 0;
            }


    .. tab-item:: malloc()

        .. code:: cpp

            #include <hip/hip_runtime.h>
            #include <iostream>

            // Addition of two values.
            __global__ void add(int* a, int* b, int* c) {
                *c = *a + *b;
            }

            int main() {
                int* a, * b, * c;

                // Allocate memory for a, b, and c.
                a = (int*)malloc(sizeof(*a));
                b = (int*)malloc(sizeof(*b));
                c = (int*)malloc(sizeof(*c));

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
                free(a);
                free(b);
                free(c);

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

                // Allocate device copies of a, b and c.
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


Missing features
================


List of HIP Managed Memory Allocation API
=========================================

.. doxygenfunction:: hipMallocManaged

.. doxygengroup:: MemoryM
   :content-only:

.. doxygenfunction:: hipPointerSetAttribute

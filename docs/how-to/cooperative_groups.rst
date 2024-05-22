.. meta::
  :description: This chapter describe how to use cooperative groups in HIP
  :keywords: AMD, ROCm, HIP, cooperative groups

*******************************************************************************
Cooperative Groups in HIP
*******************************************************************************

The Cooperative Groups is an extension of the existing ROCm programming model, 
to get a more flexible grouping mechanism for the Developers. This feature was 
introduced with CUDA 9 at NVIDIA GPUs and with ROCm 4.1 at AMD GPUs.

The API accessible in the ``cooperative_groups`` namespace after the 
``cooperative_groups.h`` is included. The header contains the following 
elements:

* Data types for representing groups
* Operations to generate implicit groups defined;
* Collectives for partitioning existing groups into new groups;
* Operation to synchronize all threads within the group;
* Operations to inspect the group properties;
* Collectives that expose low-level, group-specific and often HW accelerated, operations.

The code difference to the original block model can be found in the following table. 

.. list-table:: Cooperative Group Example
    :header-rows: 1
    :widths: 50,50

    * - **Original Block**
      - **Cooperative Groups**

    * - .. code-block:: C++
          
          __device__ int reduce_sum(int *shared, int val) {
              
              // Thread ID
              const unsigned int thread_id = threadIdx.x;

              // Every iteration the number of active threads halves,
              // until we processed all values
              for(unsigned int i = blockDim.x / 2; i > 0; i /= 2) {
                // Store value in shared memroy with thread ID
                shared[thread_id] = val;

                // Synchronize all threads
                __syncthreads();

                // Active thread sum up
                if(thread_id < i)
                    val += shared[thread_id + i];

                // Synchronize all threads in the group
                g.sync();
              }

              // ...
          }

      - .. code-block:: C++

          __device__ int reduce_sum(thread_group g, int *shared, int val) {

            // Thread ID
            const unsigned int group_thread_id = g.thread_rank();

            // Every iteration the number of active threads halves,
            // until we processed all values
            for(unsigned int i = g.size() / 2; i > 0; i /= 2) {
              // Store value in shared memroy with thread ID
              shared[group_thread_id] = val;

              // Synchronize all threads in the group
              g.sync();

              // Active thread sum up
              if(group_thread_id < i)
                val += shared[group_thread_id + i];

              // Synchronize all threads in the group
              g.sync();
            }

            // ...
          }

    * - .. code-block:: C++

          __global__ void sum_kernel(...) {
            // ...
    
            // Workspace array in shared memory
            __shared__ unsigned int workspace[2048];

            // ...


            // Perform reduction
            output = reduce_sum(workspace, input);

            // ...
          }

      - .. code-block:: C++

          __global__ void sum_kernel(...) {
            // ...

            // Workspace array in shared memory
            __shared__ unsigned int workspace[2048];

            // ...

            thread_block thread_block_group = this_thread_block();
            // Perform reduction
            output = reduce_sum(thread_block_group, workspace, input);

            // ...
          }

The kernel launch also different at cooperative groups case, which depends on the 
group type. For example, at grid groups with single GPU case the ``hipLaunchCooperativeKernel``
has to be used.

Group Types
=============

There are different group types based on different levels of grouping.

Thread Block Group
--------------------

Represents an intra-workgroup cooperative group type where the
participating threads within the group are exactly the same threads
which are participated in the currently executing ``workgroup``.

.. code-block:: C++
  
  class thread_block;

Constructed via:

.. code-block:: C++
  
  thread_block g = this_thread_block();

The ``group_index()`` , ``thread_index()`` , ``thread_rank()`` , ``size()``, ``cg_type()``,  
``is_valid()`` , ``sync()`` and ``group_dim()`` member functions are public of the 
thread_block class. For further details check the :ref:`thread_block references <thread_block_ref>` . 

Grid Group
------------

Represents an inter-workgroup cooperative group type where the participating threads
within the group spans across multiple workgroups running the (same) kernel on the same device.
To be able to synchronize across the grid, you need to use the cooperative launch API.

.. code-block:: C++

  class grid_group;

Constructed via:

.. code-block:: C++

  grid_group g = this_grid();

The ``thread_rank()`` , ``size()``, ``cg_type()``, ``is_valid()`` and ``sync()`` member functions
are public of the ``grid_group`` class. For further details check the :ref:`grid_group references <grid_group_ref>`. 

Multi Grid Group
------------------

Represents an inter-device cooperative group type where the participating threads
within the group spans across multiple devices, running the (same) kernel on these devices
All the mutli grid group APIs require that you have used the appropriate launch API.

.. code-block:: C++

  class multi_grid_group;

Constructed via:

.. code-block:: C++

  // Kernel must be launched with the cooperative multi-device API
  multi_grid_group g = this_multi_grid();

The ``num_grids()`` , ``grid_rank()`` , ``thread_rank()``, ``size()``, ``cg_type()``, ``is_valid()`` ,
and ``sync()`` member functions are public of the ``multi_grid_group`` class. For
further details check the :ref:`multi_grid_group references <multi_grid_group_ref>` . 

Thread Block Tile
------------------

This constructs a templated class derived from ``thread_group``. The template defines tile
size of the new thread group at compile time.

.. code-block:: C++

  template <unsigned int Size, typename ParentT = void>
  class thread_block_tile;

Constructed via:

.. code-block:: C++
  
  template <unsigned int Size, typename ParentT>
  _CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)


.. note::
  
  * ``Size`` must be a power of 2 and not bigger than wavefront size.
  * ``shfl`` functions are currently supports integer or float type.

The ``thread_rank()`` , ``size()``, ``cg_type()``, ``is_valid()``, ``sync()``, 
``meta_group_rank()``, ``meta_group_size()``, ``shfl(...)``, ``shfl_down(...)``, 
``shfl_up(...)`` and ``shfl_xor(...)`` member functions are public of the ``thread_block_tile``
class. For further details check the :ref:`thread_block_tile references <thread_block_tile_ref>` . 

Coalesced Groups
------------------

Represents an active thread group in a wavefront. This group type also supports sub-wave level
intrinsics.

.. code-block:: C++

  class coalesced_group;

Constructed via:

.. code-block:: C++

  coalesced_group active = coalesced_threads();

.. note::
  
  * ``shfl`` functions are currently supports integer or float type.

The ``thread_rank()`` , ``size()``, ``cg_type()``, ``is_valid()``, ``sync()``, 
``meta_group_rank()``, ``meta_group_size()``, ``shfl(...)``, ``shfl_down(...)``, 
and ``shfl_up(...)`` member functions are public of the ``coalesced_group`` class. 
For further details check the :ref:`coalesced_group references <coalesced_group_ref>` .

Synchronization
=================

At different type of gourps the synchronization requires to used the correct cooperative group
launch API.

Thread-Block Synchronization
-----------------------------------------------

1. The new block representation can be accessed with the original kernel launch methods.
2. The device side synchronization is written in the following form.

.. code-block:: C++

  thread_block g = this_thread_block();
  g.sync();

Grid Synchronization
---------------------

This section describes the necessary step to be able to syncronize group over a single GPU:

1. Check the cooperative launch capabality on single AMD GPU:

.. code-block:: C++

    int device               = 0;
    int supports_coop_launch = 0;
    // Check support
    // Use hipDeviceAttributeCooperativeMultiDeviceLaunch when launching across multiple devices
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(
        hipDeviceGetAttribute(&supports_coop_launch, hipDeviceAttributeCooperativeLaunch, device));
    if(!supports_coop_launch)
    {
        std::cout << "Skipping, device " << device << " does not support cooperative groups"
                  << std::endl;
        return 0;
    }

2. Launch the cooperative kernel on single GPU:

.. code-block:: C++

    void* params[] = {&d_vector, &d_block_reduced, &d_partition_reduced};
    // Launching kernel from host.
    HIP_CHECK(hipLaunchCooperativeKernel(vector_reduce_kernel<partition_size>,
                                         dim3(num_blocks),
                                         dim3(threads_per_block),
                                         params,
                                         0,
                                         hipStreamDefault));


3. The device side synchronization over the single GPU:

.. code-block:: C++

  grid_group grid = this_grid();
  grid.sync();

Multi-Grid Synchronization
-----------------------------

This section describes the necessary step to be able to syncronize group over multiple GPU:

1. Check the cooperative launch capabality over the multiple GPUs:

.. code-block:: C++
  
  #ifdef __HIP_PLATFORM_AMD__
    int device               = 0;
    int supports_coop_launch = 0;
    // Check support
    // Use hipDeviceAttributeCooperativeMultiDeviceLaunch when launching across multiple devices
    for (int i = 0; i < numGPUs; i++) {
      HIP_CHECK(hipGetDevice(&device));
      HIP_CHECK(
          hipDeviceGetAttribute(
            &supports_coop_launch, 
            hipDeviceAttributeCooperativeMultiDeviceLaunch, 
            device));
      if(!supports_coop_launch)
      {
          std::cout << "Skipping, device " << device << " does not support cooperative groups"
                    << std::endl;
          return 0;
      }
    }
  #endif

2. Launch the cooperative kernel on single GPU:

.. code-block:: C++

    void* params[] = {&d_vector, &d_block_reduced, &d_partition_reduced};
    // Launching kernel from host.
    HIP_CHECK(hipLaunchCooperativeKernel(vector_reduce_kernel<partition_size>,
                                         dim3(num_blocks),
                                         dim3(threads_per_block),
                                         params,
                                         0,
                                         hipStreamDefault));

3. The device side synchronization over the multiple GPU:

.. code-block:: C++

  multi_grid_group multi_grid = this_multi_grid();
  multi_grid.sync();

Unsupported CUDA features
===========================

The following CUDA optional headers are not supported on HIP:

.. code-block:: C++

    // Optionally include for memcpy_async() collective
    #include <cooperative_groups/memcpy_async.h>
    // Optionally include for reduce() collective
    #include <cooperative_groups/reduce.h>
    // Optionally include for inclusive_scan() and exclusive_scan() collectives
    #include <cooperative_groups/scan.h>

The following CUDA classes in ``cooperative_groups`` namespace are not supported on HIP:

* ``cluster_group``

The following CUDA functions/operators in ``cooperative_groups`` namespace are not supported on HIP:

* ``synchronize`` 
* ``memcpy_async``
* ``wait`` and ``wait_prior``
* ``barrier_arrive`` and ``barrier_wait``
* ``invoke_one`` and ``invoke_one_broadcast``
* ``reduce``
* ``reduce_update_async`` and ``reduce_store_async``
* Reduce operators ``plus`` , ``less`` , ``greater`` , ``bit_and`` , ``bit_xor`` and ``bit_or``
* ``inclusive_scan`` and ``exclusive_scan``

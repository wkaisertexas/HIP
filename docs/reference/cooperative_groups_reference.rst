.. meta::
  :description: This chapter lists new types and device API wrappers related to Cooperative Group
                feature, which the programmer can directly use in his kernel(s) in order to
                make use of this feature.
  :keywords: AMD, ROCm, HIP, cooperative groups

*******************************************************************************
Cooperative Groups
*******************************************************************************

The following functions are located in the https://github.com/ROCm/clr repository.

.. doxygenfunction:: cooperative_groups::this_multi_grid

.. doxygenfunction:: cooperative_groups::this_grid

.. doxygenfunction:: cooperative_groups::this_thread_block

.. doxygenfunction:: cooperative_groups::coalesced_threads

.. doxygenfunction:: cooperative_groups::tiled_partition(const ParentCGTy &g)

.. doxygenfunction:: cooperative_groups::tiled_partition(const thread_group &parent, unsigned int tile_size)

.. doxygenfunction:: cooperative_groups::group_size

.. doxygenfunction:: cooperative_groups::thread_rank

.. doxygenfunction:: cooperative_groups::is_valid

.. doxygenfunction:: cooperative_groups::sync

.. _thread_block_ref:

.. doxygenclass:: cooperative_groups::thread_block
   :members:

.. _grid_group_ref:

.. doxygenclass:: cooperative_groups::grid_group
   :members:

.. _multi_grid_group_ref:

.. doxygenclass:: cooperative_groups::multi_grid_group
   :members:
   
.. _thread_block_tile_ref:

.. doxygenclass:: cooperative_groups::thread_block_tile
   :members:

.. _coalesced_group_ref:

.. doxygenclass:: cooperative_groups::coalesced_group
   :members:

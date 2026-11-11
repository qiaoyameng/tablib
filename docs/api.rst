.. _api:

===
API
===


.. module:: tablib

This part of the documentation covers all the interfaces of Tablib.  For
parts where Tablib depends on external libraries, we document the most
important right here and provide links to the canonical documentation.


--------------
Dataset Object
--------------


.. autoclass:: Dataset
   :inherited-members:


---------------
Databook Object
---------------


.. autoclass:: Databook
   :inherited-members:


------------------
Snapshots & Views
------------------

A :class:`~tablib.Snapshot` is an immutable capture of a :class:`Dataset`
that shares immutable scalar storage through storage blocks. Overlapping
:class:`~tablib.DatasetView` objects read through to the same blocks, while
:meth:`DatasetView.to_branch` materializes an editable
:class:`~tablib.SnapshotDataset` using copy-on-write.


.. autoclass:: Snapshot
   :members:

.. autoclass:: DatasetView
   :members:

.. autoclass:: SnapshotDataset
   :members:

.. autoclass:: StorageBlock
   :members:

.. autoclass:: StorageStats
   :members:

.. autofunction:: is_immutable_scalar


---------
Functions
---------


.. autofunction:: detect_format

.. autofunction:: import_set


----------
Exceptions
----------


.. automodule:: tablib.exceptions
   :members:


Now, go start some :ref:`Tablib Development <development>`.


BrainMaze: Brain Electrophysiology, Behavior and Dynamics Analysis Toolbox - Utils
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This toolbox provides a generic tools for the BrainMaze package. This tool was separated from the BrainMaze toolbox to support a convenient and lightweight sharing of these tools across projects.

This project was originally developed as a part of the `BEhavioral STate Analysis Toolbox (BEST) <https://github.com/bnelair/best-toolbox>`_ project. However, the development has transferred to the BrainMaze project.


Documentation
"""""""""""""""

Documentation is available `here <https://bnelair.github.io/brainmaze_utils>`_.


Installation
"""""""""""""""""""""""""""

.. code-block:: bash

    pip install brainmaze-utils

How to contribute
"""""""""""""""""""""""""""
The project has 2 main protected branches *main* that contains official software releases and *dev* that contains the latest feature implementations shared with developers.
To implement a new feature a new branch should be created from the *dev* branch with name pattern of *developer_identifier/feature_name*.

After the feature is implemented, a pull request can be created to merge the feature branch into the *dev* branch with. Pull requests need to be reviewed by the code owners.

Releasing
'''''''''''''''''''''''''''''''

Releases are automated and never bypass branch protection. To cut a release, a code owner runs the **Prepare release** GitHub Action (*Actions* tab, ``workflow_dispatch``) and selects the bump (*patch* / *minor* / *major*). This opens a small ``Release vX.Y.Z`` pull request that bumps ``[project].version`` in ``pyproject.toml`` on a ``release/bump-*`` branch off *main*. Once a code owner approves and merges that pull request into *main*, the **Release** workflow tags the version, builds the distributions, publishes to PyPI, and drafts the GitHub release automatically. The build reads the version from ``pyproject.toml``, which remains the single source of truth.

Promotion of features from *dev* to *main* is independent of releases and **must not change** ``[project].version`` -- a *Version guard* CI check fails any pull request outside the release flow that edits it. The version line is owned solely by the release automation on *main*; this is what keeps ``dev`` -> ``main`` merges free of version conflicts under the squash-merge policy (a version edited on both branches would otherwise conflict every release cycle).

The **Prepare release** action requires the repository/organization setting *Allow GitHub Actions to create and approve pull requests* to be enabled, so it can open the bump pull request.

New functions need to be implemented with Sphinx compatible docstrings. The documentation is automatically generated from the docstrings using Sphinx.

Building Documentation
''''''''''''''''''''''''''''''

Documentation source is in ``docs_src/`` and generated HTML output goes to ``docs/``. The ``docs/`` directory is excluded from version control (added to .gitignore) to keep the repository clean. To build documentation locally:

.. code-block:: bash

    # Install documentation dependencies
    pip install -r docs_src/requirements.txt
    
    # Build documentation
    sphinx-build -b html docs_src/source docs
    
    # Or use the provided script
    ./make_docs.sh

The generated documentation can be hosted separately (e.g., GitHub Pages, ReadTheDocs) without committing build artifacts to the repository.


License
""""""""""""""""""

This software is licensed under BSD-3Clause license. For details see the `LICENSE <https://github.com/bnelair/brainmaze_utils/blob/master/LICENSE>`_ file in the root directory of this project.


Acknowledgment
"""""""""""""""""""""""""""
This code was developed and originally published for the first time with by (Mivalt 2022, and Sladky 2022).
We appreciate you citing these papers when utilizing this toolbox in further research work.

 | F. Mivalt et V. Kremen et al., “Electrical brain stimulation and continuous behavioral state tracking in ambulatory humans,” J. Neural Eng., vol. 19, no. 1, p. 016019, Feb. 2022, doi: 10.1088/1741-2552/ac4bfd.
 |
 | V. Sladky et al., “Distributed brain co-processor for tracking spikes, seizures and behaviour during electrical brain stimulation,” Brain Commun., vol. 4, no. 3, May 2022, doi: 10.1093/braincomms/fcac115.



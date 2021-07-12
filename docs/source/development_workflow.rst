Development workflow
=====================

GitHub workflow
----------------

1. Fork the ``GADMA`` repository on the GitHub and clone a local copy.
2. Configure your local repository with an upstream remote:

    - List the current configured remote repository of the fork::

        $ git remote -v
        > origin  https://github.com/YOUR_USERNAME/GADMA.git (fetch)
        > origin  https://github.com/YOUR_USERNAME/GADMA.git (push)

    - Specify a new remote upstream repository that will be synced with the fork::

        $ git remote add upstream https://github.com/ctlab/GADMA.git

    - Verify the new upstream with::

        $ git remote -v
        > origin    https://github.com/YOUR_USERNAME/GADMA.git (fetch)
        > origin    https://github.com/YOUR_USERNAME/GADMA.git (push)
        > upstream  https://github.com/ctlab/GADMA.git (fetch)
        > upstream  https://github.com/ctlab/GADMA.git (push)

3. Usually go to the ``devel`` branch if you want to see the latest development version of GADMA (if not then change name to ``master``)::

    $ git checkout devel

or create new feature branch::

    $ git checkout upstream/devel
    $ git checkout -b feature_branch_name

Rebasing
_________

- If some commits were added to the remote repository and you want to update your local fork.
- When some of the commits are small or messy and should be squashed with some previous commits.

To do rebasing against latest ``upstream/devel`` branch::

    $ git fetch upstream
    $ git rebase -i upstream/devel

This starts up editor showing something like::

    pick 6cabe03 Make readin to be independent of checks
    pick c7afcf5 Fix codestyle
    pick 3c6906a Fix tests for read_data
    pick 7921ad3 Fix errors

    # Rebase ac75966..7921ad3 onto ac75966 (4 commands)
    #
    # Commands:
    # p, pick = use commit
    # r, reword = use commit, but edit the commit message
    # e, edit = use commit, but stop for amending
    # s, squash = use commit, but meld into previous commit
    # f, fixup = like "squash", but discard this commit's log message
    # x, exec = run command (the rest of the line) using shell
    # d, drop = remove commit
    #
    # These lines can be re-ordered; they are executed from top to bottom.
    #
    # If you remove a line here THAT COMMIT WILL BE LOST.
    #
    # However, if you remove everything, the rebase will be aborted.
    #
    # Note that empty commits are commented out

To squash the last four commits you should edit like::

    pick 6cabe03 Make readin to be independent of checks
    s c7afcf5 Fix codestyle
    s 3c6906a Fix tests for read_data
    s 7921ad3 Fix errors

    # Rebase ac75966..7921ad3 onto ac75966 (4 commands)
    #
    # Commands:
    # p, pick = use commit
    # r, reword = use commit, but edit the commit message
    # e, edit = use commit, but stop for amending
    # s, squash = use commit, but meld into previous commit
    # f, fixup = like "squash", but discard this commit's log message
    # x, exec = run command (the rest of the line) using shell
    # d, drop = remove commit
    #
    # These lines can be re-ordered; they are executed from top to bottom.
    #
    # If you remove a line here THAT COMMIT WILL BE LOST.
    #
    # However, if you remove everything, the rebase will be aborted.
    #
    # Note that empty commits are commented out

After editing we save and close opened file. Then git will ask about commit message::

    # This is a combination of 4 commits.
    # This is the 1st commit message:
    
    Make readin to be independent of checks
    
    # This is the commit message #2:
    
    Fix codestyle
    
    # This is the commit message #3:
    
    Fix tests for read_data
    
    # This is the commit message #4:
    
    Fix errors
    
    # Please enter the commit message for your changes. Lines starting
    # with '#' will be ignored, and an empty message aborts the commit.
    #
    # Date:      Tue Jun 29 22:16:34 2021 +0300
    #
    # interactive rebase in progress; onto ac75966
    # Last commands done (4 commands done):
    #    squash 3c6906a Fix tests for read_data
    #    squash 7921ad3 Fix errors
    # No commands remaining.
    # You are currently rebasing branch 'devel' on 'ac75966'.
    #
    # Changes to be committed:
    #   modified:   gadma/data/__init__.py
    #   new file:   gadma/data/data_utils.py
    #   modified:   gadma/engines/dadi_moments_common.py
    #   modified:   gadma/engines/engine.py
    #   modified:   gadma/utils/__init__.py
    #   modified:   gadma/utils/utils.py

We can put new message like::

    This is new message for squashed commit
    
    # Please enter the commit message for your changes. Lines starting
    # with '#' will be ignored, and an empty message aborts the commit.
    #
    # Date:      Tue Jun 29 22:16:34 2021 +0300
    #
    # interactive rebase in progress; onto ac75966
    # Last commands done (4 commands done):
    #    squash 3c6906a Fix tests for read_data
    #    squash 7921ad3 Fix errors
    # No commands remaining.
    # You are currently rebasing branch 'devel' on 'ac75966'.
    #
    # Changes to be committed:
    #   modified:   gadma/data/__init__.py
    #   new file:   gadma/data/data_utils.py
    #   modified:   gadma/engines/dadi_moments_common.py
    #   modified:   gadma/engines/engine.py
    #   modified:   gadma/utils/__init__.py
    #   modified:   gadma/utils/utils.py

After saving and closing we will get::

    [detached HEAD e002a41]  This is new message for squashed commit
     Date: Tue Jun 29 22:16:34 2021 +0300
     8 files changed, 171 insertions(+), 107 deletions(-)
     create mode 100644 gadma/data/data_utils.py
    Successfully rebased and updated refs/heads/devel.

Finally we **force** push update with ``-f`` flag::

    $ git push -f origin feature_branch_name

Continuous integration (CI)
----------------------------

Continuous integration in GADMA uses `GitHub Actions <https://docs.github.com/en/actions>`__ and run the following check as soon as pull request in submitted:

- checks that proposed changes conform to style guidelines (lint checks),
- run test suite and send coverage of code to `codecov <https://about.codecov.io/>`__.
- build documentation
- publish new version to `TestPyPi <https://test.pypi.org/project/gadma/>`__ (only for ``master`` branch)

As soon as new tag appears in ``master`` branch, e.g. new release is published, new version of GADMA is pushed to   `PyPi <https://pypi.org/project/gadma/>`__.

All workflow files for GitHub Actions are located in ``.github/workflows/`` directory.

Lint checks
___________

The following check is run during the linting process::

    $ pycodestyle gadma

Also another tool to check is flake::

    $ flake8 --exit-zero ./gadma

Test suite
_____________

GADMA has unittests located in ``tests`` directory.

To run test suite in local repository::

    $ pytest -v tests --disable-warnings

Flag ``--disable-warnings`` ignores warnings in output of tests. As GADMA has a lot of warnings then output is more clear when flag is set. In original test suite (for GitHub Actions) there is upper bound on time of one test run ``--timeout=400``, it could be ignored in local run but mind that it will be used on GitHub.

To get stdout of tests add ``-s`` flag::

    $ pytest -vs tests --disable-warnings

To run test that has name ``test_some_name_of_test``::

    $ pytest -vs tests -k "test_some_name_of_test" --disable-warnings

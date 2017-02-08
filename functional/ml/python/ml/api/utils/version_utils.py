

import sklearn
from distutils import version


def gte_sklearn_version(version_minimum):
    version_installed = version.LooseVersion(sklearn.__version__)
    version_minimum = version.LooseVersion(version_minimum)
    return version_installed >= version_minimum

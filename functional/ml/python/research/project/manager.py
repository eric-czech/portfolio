

import pickle
import os
import logging
from py_utils import io_utils
logger = logging.getLogger(__name__)


class ProjectManager(object):
    """ Research Project Model

    This class adds a convenience over research workflows that often operate over two main types of data:
     1. Data too big to fit in source control
     2. Data NOT too big for source control

     This class refers to them as the "data" directory and "project" directories, respectively
    """

    def __init__(self, data_dir, proj_dir, mkdirs=True):
        """
        Initialize Data Manager
        :param data_dir: Location of data files for project on disk
        :param proj_dir: Location of project files on disk
        :param mkdirs: Create project and data directories if they do not exist
        """
        self.data_dir = data_dir
        self.proj_dir = proj_dir

        # Create directories (not recursively though) if they do not already exist
        if mkdirs:
            for path in [data_dir, proj_dir]:
                if not os.path.exists(path):
                    os.mkdir(path)

    def file(self, datatype, dataset, ext, create_dir=True):
        path = os.path.join(self.data_dir, datatype)

        # Create the parent directory for the file if it does not exist
        if create_dir and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if ext is not None:
            path = os.path.join(path, '{}.{}'.format(dataset, ext))
        else:
            path = os.path.join(path, dataset)

        return path

    def save(self, datatype, dataset, d):
        path = self.file(datatype, dataset, 'pkl')
        # logger.debug('Saving data to location "{}"'.format(path))
        return io_utils.to_pickle(d, path)

    def exists(self, datatype, dataset):
        path = self.file(datatype, dataset, 'pkl')
        return os.path.exists(path)

    def load(self, datatype, dataset, loader=None):
        path = self.file(datatype, dataset, 'pkl')

        if loader is not None and not self.exists(datatype, dataset):
            logger.debug('Generating data to be saved at location "{}"'.format(path))
            self.save(datatype, dataset, loader())

        # logger.debug('Loading saved data from location "{}"'.format(path))
        return io_utils.from_pickle(path)

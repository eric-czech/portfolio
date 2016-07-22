

import pickle
import os
import logging
logger = logging.getLogger(__name__)


class ProjectManager(object):
    """ Research Project Model

    This class adds a convenience over research workflows that often operate over two main types of data:
     1. Data too big to fit in source control
     2. Data NOT too big for source control

     This class refers to them as the "data" directory and "project" directories, respectively
    """

    def __init__(self, data_dir, proj_dir):
        """
        Initialize Data Manager
        :param data_dir: Location of data files for project on disk
        :param proj_dir: Location of project files on disk
        """
        self.data_dir = data_dir
        self.proj_dir = proj_dir

        # Create directories (not recursively though) if they do not already exist
        for path in [data_dir, proj_dir]:
            if not os.path.exists(path):
                os.mkdir(path)

    def _get_data_file(self, datatype, dataset, ext):
        path = os.path.join(self.data_dir, datatype)
        path = os.path.join(path, '{}.{}'.format(dataset, ext))
        return path

    def save(self, datatype, dataset, d):
        path = self._get_data_file(datatype, dataset, 'pkl')
        logger.debug('Saving data to location "{}"'.format(path))
        with open(path, 'wb') as fd:
            pickle.dump(d, fd)
        return path

    def exists(self, datatype, dataset):
        path = self._get_data_file(datatype, dataset, 'pkl')
        return os.path.exists(path)

    def load(self, datatype, dataset, raise_on_absent=True):
        path = self._get_data_file(datatype, dataset, 'pkl')
        logger.debug('Loading saved data from location "{}"'.format(path))
        if not raise_on_absent and not os.path.exists(path):
            return None
        with open(path, 'rb') as fd:
            return pickle.load(fd)

# Utilities for processing zip archives

import zipfile


def _validate_archive(zip_filepath):
    assert zip_filepath.endswith('.zip'), \
        'File should have a .zip extension (filepath given = "{}")'.format(zip_filepath)


def get_zip_archive_files(zip_filepath):
    _validate_archive(zip_filepath)
    with zipfile.ZipFile(zip_filepath, "r") as zfile:
        return zfile.namelist()


def get_zip_archive_file_data(zip_filepath, filename):
    _validate_archive(zip_filepath)
    with zipfile.ZipFile(zip_filepath, "r") as zfile:
        filenames = zfile.namelist()
        assert filename in filenames, \
            'Failed to find file "{}" in zip archive "{}" (filenames available = "{}")'\
            .format(filename, zip_filepath, filenames)
        return zfile.read(filename)

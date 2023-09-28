""" Python script to validate data

Run as:

    python3 scripts/validate_data.py data
"""

from pathlib import Path
import sys
import os
import hashlib


def file_hash(filename):
    """ Get byte contents of file `filename`, return SHA1 hash

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    hash : str
        SHA1 hexadecimal hash string for contents of `filename`.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        for line in f:
            sha1.update(line)
    return sha1.hexdigest()

    raise NotImplementedError(
        'This is just a template -- you are expected to code this.')


def validate_data(data_directory):
    """ Read ``data_hashes.txt`` file in `data_directory`, check hashes

    Parameters
    ----------
    data_directory : str
        Directory containing data and ``data_hashes.txt`` file.

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If hash value for any file is different from hash value recorded in
        ``data_hashes.txt`` file.
    """
    # debug print statement
    print(f"Looking for 'hash_list.txt' in directory: {data_directory}")

    hash_list_path = os.path.join(data_directory, 'group-00', 'hash_list.txt')

    # another debug print statement
    print(f"Full path to 'hash_list.txt' found: {hash_list_path}")

    # open file and read each line
    try:
        with open(hash_list_path, 'rt') as f:
            lines = f.readlines()
    # raising error if the file was not found
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File 'hash_list.txt' not found in directory: {data_directory}/group-00")


    #rading each line of the file
    for line in lines:
        expected_hash, filename = line.split()
        actual_hash = file_hash(os.path.join(data_directory, filename))

        if expected_hash != actual_hash:
            raise ValueError(f"Hash for {filename} does not match")


    # raise NotImplementedError('This is just a template -- you are expected to code this.')

def main():
    # This function (main) called when this file run as a script.
    #
    # Get the data directory from the command line arguments
    if len(sys.argv) < 2:
        raise RuntimeError("Please give data directory on "
                           "command line")
    data_directory = sys.argv[1]
    # Call function to validate data in data directory
    validate_data(data_directory)


if __name__ == '__main__':
    # Python is running this file as a script, not importing it.
    main()

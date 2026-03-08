"""Hybrid evaluation pipeline: function made with the help of ChatGPT"""

import json
import shutil

def save_image(src, dst):
    """
    Move an image file from a source path to a destination path.

    This function is typically used to relocate selected images
    (e.g., filtered or classified images) into a target directory.

    Parameters
    ----------
    src : str
        Path to the source image file.
    dst : str
        Destination path where the image will be moved.
    """
    shutil.move(src, dst)

def update_metadata(meta_dict, filename, data):
    """
    Update a metadata dictionary with information for a given file.

    The function inserts or overwrites metadata associated with
    a specific filename.

    Parameters
    ----------
    meta_dict : dict
        Dictionary storing metadata for multiple files.
    filename : str
        Name of the file used as the metadata key.
    data : dict
        Metadata information to associate with the file.
    """
    meta_dict[filename] = data

def write_json(path, data):
    """
    Write a Python object to a JSON file.

    The data is serialized and written with indentation for
    improved human readability.

    Parameters
    ----------
    path : str
        File path where the JSON output will be saved.
    data : dict
        Data structure to serialize and write to disk.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
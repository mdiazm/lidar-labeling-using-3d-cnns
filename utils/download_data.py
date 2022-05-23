"""
Functions to download data from a website
"""
import requests
import os
from tqdm import tqdm
from py7zr import SevenZipFile


def download(url, savepath):
    """
    Download a file from the given URL and store it on the savepath path.

    :param url: where to download the file from
    :param savepath: where to download the file on the local filesystem (directory)
    """

    # Get filename
    filename = url.split("/")[-1]

    # Create request
    req = requests.get(url, stream=True)

    # File size
    total_file_size = int(req.headers.get('content-length', 0))
    chunk_size = 1024  # bytes, 1 KibiByte

    # Create progress bar
    progress_bar = tqdm(total=total_file_size, unit='iB', unit_scale=True)

    # Download file sequentially
    print("Downloading file {} in {}".format(filename, savepath))
    savepath = os.path.join(savepath, filename)
    with open(savepath, 'wb') as fd:
        for chunk in req.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fd.write(chunk)

    # Close progress bar
    progress_bar.close()

    # Check for errors
    if total_file_size != 0 and progress_bar.n != total_file_size:
        print("Something was wrong whilst downloading file {}".format(filename))

def unzip(orig_file, savepath):
    """
    Unzip 7z file using py7zr library.

    :param orig_file: path of the original compressed 7z file
    :param savepath: where to save the uncompressed file
    """

    filename = orig_file.split(os.path.sep)[-1]

    print("Decompressing {} file and storing it in {}...".format(filename, savepath))

    with SevenZipFile(orig_file, 'r') as cf:
        cf.extractall(path=savepath)


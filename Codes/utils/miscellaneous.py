import os

def get_filename(file_path: str) -> str:
    """
    Extract the file name and file extension.
    """

    # Get the filename without extension
    file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]

    # Get the file extension
    file_extension = os.path.splitext(file_path)[1]

    return [file_name_without_extension, file_extension]

def isepoch(
    epoch,
    per,
):
    return epoch % per == (per - 1)

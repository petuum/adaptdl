from enum import Enum
import os


# Adapted from Ray Tune
def _checkpoint_obj_to_dir(checkpoint_dir, checkpoint_obj):
    for (path, data) in checkpoint_obj.items():
        file_path = os.path.join(checkpoint_dir, path)
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(data)
    return


def _serialize_checkpoint(checkpoint_dir):
    data = {}
    for basedir, _, file_names in os.walk(checkpoint_dir):
        for file_name in file_names:
            path = os.path.join(basedir, file_name)
            with open(path, "rb") as f:
                data[os.path.relpath(path, checkpoint_dir)] = f.read()
    return data


class Status(Enum):
    FAILED = 0
    SUCCEEDED = 1
    RUNNING = 2

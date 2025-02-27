import yadisk
import posixpath
import shutil
import os


def recursive_upload(client: yadisk.Client, from_dir: str, to_dir: str):
    for root, dirs, files in os.walk(from_dir):
        p = root.split(from_dir)[1].strip(os.path.sep)
        dir_path = posixpath.join(to_dir, p)

        try:
            client.mkdir(dir_path)
        except yadisk.exceptions.PathExistsError:
            pass

        for file in files:
            file_path = posixpath.join(dir_path, file)
            p_sys = p.replace("/", os.path.sep)
            in_path = os.path.join(from_dir, p_sys, file)

            try:
                client.upload(in_path, file_path)
            except yadisk.exceptions.PathExistsError:
                pass
def scandir(dir_path, suffix=None, recursive=False):

    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must ne a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(entry.path,
                                        suffix=suffix,
                                        recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

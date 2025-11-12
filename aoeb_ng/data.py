import logging
import mmap
import os
import tarfile

logger = logging.getLogger(__name__)


class IndexedMmapTarFile:
    def __init__(self, path: str, index_suffix: str = '.index'):
        self.path = path
        self.index_path = path + index_suffix
        self.index = dict()
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as file:
                for line in file:
                    key, start, length = line.strip().split('\t')
                    self.index[key] = (int(start), int(length))
        else:
            self.build_index()
        self._file = open(path, mode='r')
        self._mmap = mmap.mmap(self._file.fileno(), 0, prot=mmap.PROT_READ)
        logger.info(f"Loaded IndexedMmapTarFile from {path} with index {len(self.index)} entries.")

    def build_index(self):
        with tarfile.open(self.path, 'r') as tf, open(self.index_path, "w") as idx_file:
            for member in tf.getmembers():
                if member.isfile():
                    idx_file.write(f"{member.path}\t{member.offset_data}\t{member.size}\n")
                    self.index[member.path] = (member.offset_data, member.size)
            logger.warning(f"Index built at {self.index_path}")
        return

    def __del__(self):
        self._mmap.close()
        self._file.close()

    def __len__(self):
        return len(self.index)

    def __contains__(self, key):
        return key in self.index

    def __getitem__(self, key: str) -> bytes:
        if key not in self.index:
            raise KeyError(f"Key {key} not found in index.")
        start, length = self.index[key]
        self._mmap.seek(start)
        data_bytes = self._mmap.read(length)
        return data_bytes


if __name__ == "__main__":
    tar_path = "py.tar"
    indexed_tar = IndexedMmapTarFile(tar_path)
    print(f"Number of entries: {len(indexed_tar)}")
    for key in indexed_tar.index.keys():
        data = indexed_tar[key]
        print(f"Read {len(data)} bytes for key {key}")

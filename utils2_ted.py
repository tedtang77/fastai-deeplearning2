import klepto
from klepto.archives import file_archive, dir_archive


def save_big_data(fpath, fname, data):
    """
    https://stackoverflow.com/questions/17513036/pickle-dump-huge-file-without-memory-error
    """
    arch = dir_archive(fpath+fname, cached=False, serialized=True)
    arch[fname] = data
    # # dump from memory cache to the on-disk archive
    arch.dump()
    # clear the memory cache
    #arch.clear()
    
    
def load_big_data(fpath, fname):
    """
    https://stackoverflow.com/questions/17513036/pickle-dump-huge-file-without-memory-error
    """
    arch = dir_archive(fpath+fname, cached=False, serialized=True)
    arch.load(fname)
    return arch[fname]


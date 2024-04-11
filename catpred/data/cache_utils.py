import os
from shutil import rmtree
import torch
import hashlib
from functools import wraps
from pathlib import Path

def exists(val):
    return val is not None

# constants

CACHE_PATH = Path(os.getenv('CATPRED_CACHE_PATH', os.path.expanduser('~/.cache.esm2_embeddings')))
CACHE_PATH.mkdir(exist_ok = True, parents = True)

CLEAR_CACHE = exists(os.getenv('CLEAR_CACHE', None))
VERBOSE = exists(os.getenv('VERBOSE', None))

# helper functions

def log(s):
    if not VERBOSE:
        return
    print(s)

def md5_hash_fn(s):
    encoded = s.encode('utf-8')
    return hashlib.md5(encoded).hexdigest()

# run once function

GLOBAL_RUN_RECORDS = dict()

def run_once(global_id = None):
    def outer(fn):
        has_ran_local = False
        output = None

        @wraps(fn)
        def inner(*args, **kwargs):
            nonlocal has_ran_local
            nonlocal output

            has_ran = GLOBAL_RUN_RECORDS.get(global_id, False) if exists(global_id) else has_ran_local

            if has_ran:
                return output

            output = fn(*args, **kwargs)

            if exists(global_id):
                GLOBAL_RUN_RECORDS[global_id] = True

            has_ran = True
            return output

        return inner
    return outer

# caching function

def cache_fn(
    fn,
    path = '',
    name = None,
    hash_fn = md5_hash_fn,
    clear = False or CLEAR_CACHE,
    should_cache = True
):
    if not should_cache:
        return fn

    (CACHE_PATH / path).mkdir(parents = True, exist_ok = True)

    @run_once(path)
    def clear_cache_folder_():
        cache_path = rmtree(str(CACHE_PATH / path))
        (CACHE_PATH / path).mkdir(parents = True, exist_ok = True)

    @wraps(fn)
    def inner(t, *args, __cache_key = None, **kwargs):
        if clear:
            clear_cache_folder_()

        if name is None: 
            cache_str = __cache_key if exists(__cache_key) else t
            key = hash_fn(cache_str)
        else:
            key = name

        entry_path = CACHE_PATH / path / f'{key}.pt'

        if entry_path.exists():
            log(f'cache hit: fetching {t} from {str(entry_path)}')
            return torch.load(str(entry_path))

        out = fn(t, *args, **kwargs)

        log(f'saving: {t} to {str(entry_path)}')
        torch.save(out, str(entry_path))
        return out
        
    return inner

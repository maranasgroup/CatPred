import os
from shutil import rmtree
import torch
import hashlib
from functools import wraps
from pathlib import Path
from catpred.security import load_torch_artifact

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


def cache_entry_path(path='', cache_key=None, hash_fn=md5_hash_fn, name=None):
    (CACHE_PATH / path).mkdir(parents=True, exist_ok=True)
    if name is None:
        key = hash_fn(cache_key)
    else:
        key = name
    return CACHE_PATH / path / f'{key}.pt'


def load_cache_value(path='', cache_key=None, *, purpose="cache entry", hash_fn=md5_hash_fn, name=None, map_location=None):
    entry_path = cache_entry_path(path=path, cache_key=cache_key, hash_fn=hash_fn, name=name)
    if not entry_path.exists():
        return None

    log(f'cache hit: fetching {cache_key} from {str(entry_path)}')
    return load_torch_artifact(
        str(entry_path),
        purpose=purpose,
        map_location=map_location,
        roots=[CACHE_PATH],
    )


def save_cache_value(value, path='', cache_key=None, *, hash_fn=md5_hash_fn, name=None):
    entry_path = cache_entry_path(path=path, cache_key=cache_key, hash_fn=hash_fn, name=name)
    log(f'saving: {cache_key} to {str(entry_path)}')
    cache_value = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    torch.save(cache_value, str(entry_path))
    return entry_path

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

        cache_str = __cache_key if exists(__cache_key) else t
        cached = load_cache_value(
            path=path,
            cache_key=cache_str,
            purpose="esm cache entry",
            hash_fn=hash_fn,
            name=name,
        )
        if cached is not None:
            return cached

        out = fn(t, *args, **kwargs)

        save_cache_value(out, path=path, cache_key=cache_str, hash_fn=hash_fn, name=name)
        return out
        
    return inner

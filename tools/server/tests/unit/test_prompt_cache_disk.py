import os
import shutil
import re
import stat
import pytest
from utils import *

# end-to-end tests for --cache-disk: prompt cache entries evicted from RAM are written to disk and read back
# the file format itself is covered by tests/test-server-prompt-cache.cpp

# stories260K keeps about 640 bytes of KV per token, so a 400-token prompt (~256 KiB) fits in
# --cache-ram 1 and a few of them push the oldest out of RAM; flood() stops at the first eviction
N_PROMPT = 400
N_FLOOD_MAX = 64

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = make_server(tmp_path)


def make_server(tmp_path, disk=True, log_name="server.log"):
    s = ServerPreset.tinyllama2()
    s.n_slots = 1
    s.n_predict = 8
    s.temperature = 0.0
    s.cache_ram = 1
    s.cache_disk = str(tmp_path / "pcd") if disk else None
    s.debug = True
    s.log_path = str(tmp_path / log_name)
    return s


def prompt(i, n=N_PROMPT):
    # no common prefix between prompts, all ids inside the 512-token vocab
    return [20 + i] + [(i * 37 + j * 7) % 480 + 20 for j in range(1, n)]


def complete(s, tokens):
    res = s.make_request("POST", "/completion", data={
        "prompt": tokens,
        "cache_prompt": True,
        "return_tokens": True,
    })
    assert res.status_code == 200, res.body
    return res.body


def flood(s, start=1, count=None, make=prompt):
    """unrelated prompts on the one slot; with count=None, until the RAM cache has evicted its oldest entry"""
    for i in range(start, start + (count or N_FLOOD_MAX)):
        complete(s, make(i))
        if count is None and "removing oldest entry" in read_log(s):
            return
    assert count is not None, f"nothing was evicted from the RAM cache after {N_FLOOD_MAX} prompts"


def read_log(s):
    with open(s.log_path, encoding="utf-8", errors="replace") as f:
        return f.read()


def pcd_files(path):
    if not os.path.isdir(path):
        return []
    return sorted(f for f in os.listdir(path) if f.endswith(".pcd"))


def tmp_files(path):
    if not os.path.isdir(path):
        return []
    return sorted(f for f in os.listdir(path) if f.endswith(".tmp"))


def run_evict_and_hit(s):
    """P0, then enough other prompts to push P0 out of RAM, then P0 again"""
    s.start()
    first = complete(s, prompt(0))
    flood(s)
    again = complete(s, prompt(0))
    s.stop()
    return first, again, read_log(s)


def test_evicted_prompt_written_to_disk():
    server.start()
    complete(server, prompt(0))
    flood(server)
    log = read_log(server)
    assert "prompt cache disk tier: '" in log
    assert "wrote evicted prompt with" in log
    files = pcd_files(server.cache_disk)
    assert len(files) >= 1
    assert all(re.fullmatch(r"[0-9a-f]{16}-[0-9]+\.pcd", f) for f in files), files
    assert tmp_files(server.cache_disk) == []


def test_disk_hit_restores_prompt(tmp_path):
    # control: without the disk tier the evicted prompt is processed again
    _, again_ram, log_ram = run_evict_and_hit(make_server(tmp_path, disk=False, log_name="ram.log"))
    assert again_ram["timings"]["cache_n"] == 0
    assert "read prompt with" not in log_ram
    n_full = again_ram["timings"]["prompt_n"]

    _, again, log = run_evict_and_hit(server)
    assert "read prompt with" in log
    # the entry also holds the generated tokens, so it is a prefix hit, not a full restore
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in log
    assert again["timings"]["prompt_n"] <= 2
    assert again["timings"]["cache_n"] + again["timings"]["prompt_n"] == n_full


def test_output_identical_with_and_without_disk(tmp_path):
    first_ram, again_ram, _ = run_evict_and_hit(make_server(tmp_path, disk=False, log_name="ram.log"))
    first, again, log = run_evict_and_hit(server)
    assert "read prompt with" in log
    assert first["content"] == first_ram["content"]
    assert again["content"] == again_ram["content"] == first["content"]
    assert again["tokens"] == again_ram["tokens"] == first["tokens"]


def test_restart_reuses_files():
    server.start()
    first = complete(server, prompt(0))
    flood(server)
    server.stop()
    files = pcd_files(server.cache_disk)
    assert len(files) >= 1
    mtimes = {f: os.stat(os.path.join(server.cache_disk, f)).st_mtime_ns for f in files}

    server.start()
    log = read_log(server)
    assert f"prompt cache disk tier: '{server.cache_disk}', {len(files)} entries" in log
    again = complete(server, prompt(0))
    log = read_log(server)
    assert "read prompt with" in log
    assert again["timings"]["prompt_n"] <= 2
    assert again["content"] == first["content"]

    # the hit file is kept and becomes the most recent entry
    assert pcd_files(server.cache_disk) == files
    touched = [f for f in files if os.stat(os.path.join(server.cache_disk, f)).st_mtime_ns > mtimes[f]]
    assert len(touched) == 1


def test_shutdown_flush_survives_restart(tmp_path):
    # nothing is evicted from RAM: without the flush on shutdown the disk tier stays empty
    def run(name, disk):
        s = make_server(tmp_path, disk=disk, log_name=f"{name}.log")
        s.cache_ram = 64
        if disk:
            s.cache_disk = str(tmp_path / f"pcd-{name}")
        s.start()
        first = complete(s, prompt(0))
        s.stop()
        s.start()
        again = complete(s, prompt(0))
        s.stop()
        return first, again, read_log(s), s.cache_disk

    # control: a restart without the disk tier processes the prompt again
    _, lost, _, _ = run("ram", disk=False)
    assert lost["timings"]["cache_n"] == 0

    first, again, log, path = run("disk", disk=True)
    assert "wrote evicted prompt with" not in log
    assert "RAM entries to disk on shutdown" in log
    assert len(pcd_files(path)) >= 1
    assert tmp_files(path) == []
    assert "read prompt with" in log
    assert again["timings"]["prompt_n"] <= 2
    assert again["content"] == first["content"]


@pytest.mark.parametrize("variant", ["model_path", "ctk", "fa", "draft", "other_model", "rope", "mtime"])
def test_files_of_another_setup_are_ignored(tmp_path, variant):
    server.fa = "off"
    if variant == "mtime":
        # run from a private copy, so the newer mtime below does not touch the shared model cache
        server.start()
        src = os.path.abspath(server.make_request("GET", "/props").body["model_path"])
        server.stop()
        copy = tmp_path / "copy.gguf"
        shutil.copyfile(src, copy)
        server.model_hf_repo = None
        server.model_hf_file = None
        server.model_file = str(copy)
    server.start()
    model_path = os.path.abspath(server.make_request("GET", "/props").body["model_path"])  # may be relative to the cwd
    complete(server, prompt(0))
    flood(server)
    server.stop()
    files = pcd_files(server.cache_disk)
    assert len(files) >= 1

    other = make_server(tmp_path, log_name="other.log")
    other.fa = "off"
    if variant == "model_path":
        # same file under another path: a finetune can share desc, size and n_params with its base model
        if os.name == "nt":
            pytest.skip("symlinks need extra privileges on Windows")
        link = tmp_path / "link.gguf"
        os.symlink(model_path, link)
        other.model_hf_repo = None
        other.model_hf_file = None
        other.model_file = str(link)
    elif variant == "ctk":
        other.ctk = "q8_0"
    elif variant == "fa":
        other.fa = "on"
    elif variant == "draft":
        # a loaded draft model alone is not enough, speculation must be on for the draft context to exist
        other.model_draft_hf_repo = other.model_hf_repo
        other.spec_type = "draft-simple"
    elif variant == "rope":
        # a RoPE override changes the cached KV values, same model and KV types
        other.rope_freq_base = 20000
    elif variant == "mtime":
        # a new version downloaded over the same file name: same path, desc and size, newer mtime
        other.model_hf_repo = None
        other.model_hf_file = None
        other.model_file = model_path
        t = os.stat(model_path).st_mtime + 60
        os.utime(model_path, (t, t))
    elif variant == "other_model":
        other = ServerPreset.tinygemma3()
        other.cache_ram = 1
        other.cache_disk = server.cache_disk
        other.debug = True
        other.log_path = str(tmp_path / "other.log")

    other.start()
    log = read_log(other)
    assert f"prompt cache disk tier: '{server.cache_disk}', 0 entries" in log
    assert log.count("ignoring prompt cache file") == len(files)
    if variant != "other_model":
        res = complete(other, prompt(0))
        assert res["timings"]["cache_n"] == 0
    other.stop()
    assert "read prompt with" not in read_log(other)
    # files of another setup are never deleted
    assert set(files) <= set(pcd_files(server.cache_disk))


# truncated: the read runs out of data; bitflip: one bit in the middle of the target state, caught by the digest
@pytest.mark.parametrize("damage", ["truncate", "bitflip"])
def test_corrupt_file_is_deleted_and_server_keeps_serving(damage):
    server.start()
    first = complete(server, prompt(0))
    flood(server)
    server.stop()
    files = pcd_files(server.cache_disk)
    assert len(files) >= 1
    for f in files:
        path = os.path.join(server.cache_disk, f)
        size = os.path.getsize(path)
        with open(path, "r+b") as fh:
            if damage == "truncate":
                fh.truncate(size // 2)
            else:
                fh.seek(size // 2)
                b = fh.read(1)
                fh.seek(size // 2)
                fh.write(bytes([b[0] ^ 0x10]))

    # also: junk with our extension, a temporary file of an interrupted write, and a foreign .tmp file
    with open(os.path.join(server.cache_disk, "junk.pcd"), "wb") as fh:
        fh.write(os.urandom(4096))
    with open(os.path.join(server.cache_disk, "0000000000000001-0.pcd.tmp"), "wb") as fh:
        fh.write(b"partial")
    with open(os.path.join(server.cache_disk, "foreign.tmp"), "wb") as fh:
        fh.write(b"not ours")

    server.start()
    log = read_log(server)
    assert "ignoring prompt cache file" in log and "junk.pcd" in log
    assert not os.path.exists(os.path.join(server.cache_disk, "0000000000000001-0.pcd.tmp"))
    assert os.path.exists(os.path.join(server.cache_disk, "foreign.tmp"))

    again = complete(server, prompt(0))
    log = read_log(server)
    assert "failed to read prompt cache file" in log
    assert again["timings"]["cache_n"] == 0
    assert again["content"] == first["content"]
    # exactly the file that failed is gone, the others stay until they are hit
    assert len(set(files) - set(pcd_files(server.cache_disk))) == 1
    assert os.path.exists(os.path.join(server.cache_disk, "junk.pcd"))

    # still serving
    complete(server, prompt(100))
    assert server.make_request("GET", "/health").status_code == 200


def test_size_cap_deletes_oldest_first():
    server.cache_disk_size = 1  # MiB, room for a few entries
    server.start()
    flood(server, start=0, count=24)
    server.stop()
    log = read_log(server)
    deleted = [os.path.basename(p) for p in re.findall(r"prompt cache disk limit reached, deleting '([^']+)'", log)]
    kept = pcd_files(server.cache_disk)
    assert len(deleted) >= 1
    assert len(kept) >= 1
    assert sum(os.path.getsize(os.path.join(server.cache_disk, f)) for f in kept) <= 1024 * 1024
    # file names start with the write time, so every deleted file must be older than every kept one
    assert max(deleted) < min(kept)
    assert deleted == sorted(deleted)
    assert not set(deleted) & set(kept)


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions and a non-root user")
def test_read_only_directory():
    os.makedirs(server.cache_disk)
    os.chmod(server.cache_disk, stat.S_IRUSR | stat.S_IXUSR)
    try:
        server.start()
        first = complete(server, prompt(0))
        flood(server)
        again = complete(server, prompt(0))
        assert again["content"] == first["content"]
        assert again["timings"]["cache_n"] == 0
        log = read_log(server)
        assert "failed to write prompt cache file" in log
        assert pcd_files(server.cache_disk) == []
        assert tmp_files(server.cache_disk) == []
    finally:
        os.chmod(server.cache_disk, stat.S_IRWXU)


def test_missing_directory_is_created(tmp_path):
    server.cache_disk = str(tmp_path / "a" / "b" / "pcd")
    server.start()
    complete(server, prompt(0))
    flood(server)
    assert os.path.isdir(server.cache_disk)
    assert len(pcd_files(server.cache_disk)) >= 1


def test_uncreatable_directory_disables_tier(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_bytes(b"x")
    server.cache_disk = str(blocker / "pcd")
    server.start()
    first = complete(server, prompt(0))
    flood(server)
    again = complete(server, prompt(0))
    assert again["content"] == first["content"]
    log = read_log(server)
    assert "disk tier disabled" in log
    assert "wrote evicted prompt with" not in log
    assert blocker.read_bytes() == b"x"


def test_disk_tier_needs_cache_ram():
    server.cache_ram = 0
    server.start()
    complete(server, prompt(0))
    flood(server, count=8)
    assert not os.path.exists(server.cache_disk)
    assert "--cache-disk has no effect" in read_log(server)


def test_negative_size_rejected():
    server.cache_disk_size = -1
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=30)


# no tiny recurrent/hybrid model in the presets: tinygemma3 (SWA) is the closest, it cannot roll back past its window
# either and depends on context checkpoints the same way, so the checkpoints must survive the round trip through disk
def test_checkpoints_roundtrip_through_disk(tmp_path):
    base = "The quick brown fox jumps over the lazy dog. " * 20
    p_first  = base + "The first ending of this story is a happy one."
    p_second = base + "But the second ending was different and sad."

    def run(name, disk, push_out):
        s = ServerPreset.tinygemma3()
        s.temperature = 0.0
        s.n_ubatch = 32
        s.n_slots = 1
        s.cache_ram = 16  # one entry with its checkpoints is ~11 MiB
        s.cache_disk = str(tmp_path / f"pcd-{name}") if disk else None
        s.debug = True
        s.log_path = str(tmp_path / f"{name}.log")
        s.start()
        n_full = complete(s, p_first)["timings"]["prompt_n"]
        complete(s, p_second)
        if push_out:
            flood(s, make=lambda i: f"{i} " + "Unrelated words keep the slot busy for a while. " * 30)
        res = complete(s, p_first)
        s.stop()
        return n_full, res, read_log(s), s.cache_disk

    n_full, direct, _, _ = run("direct", disk=False, push_out=False)
    _, lost, _, _ = run("lost", disk=False, push_out=True)
    _, disk, log, path = run("disk", disk=True, push_out=True)

    # the scenario needs a checkpoint: the live slot does better than a full prefill
    assert direct["timings"]["prompt_n"] < n_full
    # control: pushed out of RAM without a disk tier, everything is processed again
    assert lost["timings"]["prompt_n"] == n_full
    assert "wrote evicted prompt with" in log, "entry larger than --cache-ram? then raise cache_ram here"
    assert "read prompt with" in log
    assert len(pcd_files(path)) >= 1
    # read back from disk with its checkpoints, the result matches the live slot exactly
    assert disk["timings"]["prompt_n"] == direct["timings"]["prompt_n"]
    assert disk["timings"]["cache_n"] == direct["timings"]["cache_n"]
    assert disk["content"] == direct["content"]

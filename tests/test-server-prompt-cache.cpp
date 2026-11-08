#include "testing.h"

#include "../tools/server/server-task.h"

#include "hash/xxhash/xxhash.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <new>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <sys/resource.h>
#endif

// unit tests for the disk tier of the server prompt cache (--cache-disk), no model required
// NOTE: end-to-end tests live in tools/server/tests/unit/test_prompt_cache_disk.py

namespace fs = std::filesystem;

//
// allocation tracking: a corrupted size field must never turn into a large allocation
// (the vectors use a no-init allocator, so with overcommit a huge allocation would not even fail)
//

#if defined(__SANITIZE_ADDRESS__)
#define PCD_TRACK_ALLOC 0
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer)
#define PCD_TRACK_ALLOC 0
#endif
#endif
#ifndef PCD_TRACK_ALLOC
#define PCD_TRACK_ALLOC 1
#endif

#if PCD_TRACK_ALLOC && defined(__GNUC__) && !defined(__clang__)
// GCC sees the malloc/free inside the replaced global operators and reports false positives
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

static std::atomic<bool>   g_track_alloc{false};
static std::atomic<size_t> g_max_alloc{0};

#if PCD_TRACK_ALLOC
void * operator new(std::size_t n) {
    if (g_track_alloc.load(std::memory_order_relaxed)) {
        size_t cur = g_max_alloc.load(std::memory_order_relaxed);
        while (n > cur && !g_max_alloc.compare_exchange_weak(cur, n)) {}
    }
    if (void * p = std::malloc(n ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
#endif

// largest single allocation made by f, 0 when tracking is not available
template <typename F>
static size_t max_alloc_of(F && f) {
    g_max_alloc = 0;
    g_track_alloc = true;
    f();
    g_track_alloc = false;
    return g_max_alloc;
}

// a corrupted file must not allocate more than this (file buffers, paths and the like stay far below)
static constexpr size_t ALLOC_BOUND = 1024*1024;

//
// helpers
//

static const std::string FP = "/models/test.gguf|llama 260K F16|292352|1185280|1|1|-1|0";

// on-disk layout, see the comment above PCD_MAGIC in server-task.cpp:
// magic u32, version u32, fp (u64 size + bytes), tokens (u64 size + bytes), main, drft (u64 size + bytes),
// n_ckpt u32, per checkpoint n_tokens i64, pos_min i32, pos_max i32, tgt, dft, spec (u64 size + bytes),
// then the XXH3-64 digest (u64) of everything from the tokens size field up to the digest
static constexpr size_t OFF_MAGIC   = 0;
static constexpr size_t OFF_VERSION = 4;
static constexpr size_t OFF_FP_LEN  = 8;
static constexpr size_t OFF_FP      = 16;
static constexpr size_t DIGEST_SIZE = 8;

struct tmp_dir {
    fs::path path;

    explicit tmp_dir(const std::string & name) {
        static std::atomic<int> counter{0};
        std::random_device rd;
        path = fs::temp_directory_path() / ("test-server-prompt-cache-" + name + "-" + std::to_string(rd()) + "-" + std::to_string(counter++));
        fs::remove_all(path);
        fs::create_directories(path);
    }

    ~tmp_dir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }

    std::string str() const { return path.string(); }
};

static common_state_data fill(size_t n, uint8_t seed) {
    common_state_data res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = (uint8_t) (seed + 31*i + (i >> 8));
    }
    return res;
}

static llama_tokens make_tokens(size_t n, llama_token first) {
    llama_tokens res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = first + (llama_token) i;
    }
    return res;
}

static server_prompt_cache_state make_state(const llama_tokens & tokens, size_t n_main, size_t n_drft, int n_ckpt, size_t n_spec, uint8_t seed = 1, size_t n_ckpt_tgt = 16) {
    server_prompt_cache_state st;
    st.prompt.tokens = server_tokens(tokens, false);
    st.data.main = fill(n_main, seed);
    st.data.drft = fill(n_drft, seed + 1);
    for (int i = 0; i < n_ckpt; ++i) {
        auto ckpt = std::make_shared<common_prompt_checkpoint>();
        ckpt->n_tokens  = 1000000000000ll + i; // needs all 8 bytes
        ckpt->pos_min   = -1 - i;
        ckpt->pos_max   = 100 + i;
        ckpt->data_tgt  = fill(n_ckpt_tgt + i, seed + 2 + i);
        ckpt->data_dft  = fill(n_drft ? 8 + i : 0, seed + 3 + i);
        ckpt->data_spec = std::vector<uint8_t>(n_spec, (uint8_t) (seed + i));
        st.prompt.checkpoints.push_back(std::move(ckpt));
    }
    return st;
}

static bool same_state(const server_prompt_cache_state & a, const server_prompt_cache_state & b) {
    if (a.prompt.tokens.get_text_tokens() != b.prompt.tokens.get_text_tokens() ||
        a.data.main != b.data.main || a.data.drft != b.data.drft ||
        a.prompt.checkpoints.size() != b.prompt.checkpoints.size()) {
        return false;
    }
    for (auto ia = a.prompt.checkpoints.begin(), ib = b.prompt.checkpoints.begin(); ia != a.prompt.checkpoints.end(); ++ia, ++ib) {
        const auto & ca = **ia;
        const auto & cb = **ib;
        if (ca.n_tokens != cb.n_tokens || ca.pos_min != cb.pos_min || ca.pos_max != cb.pos_max ||
            ca.data_tgt != cb.data_tgt || ca.data_dft != cb.data_dft || ca.data_spec != cb.data_spec) {
            return false;
        }
    }
    return true;
}

static std::vector<uint8_t> read_bytes(const fs::path & path) {
    std::ifstream ifs(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

static void write_bytes(const fs::path & path, const std::vector<uint8_t> & data) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    ofs.write((const char *) data.data(), data.size());
}

template <typename T>
static void poke(std::vector<uint8_t> & data, size_t off, T v) {
    memcpy(data.data() + off, &v, sizeof(v));
}

template <typename T>
static T peek(const std::vector<uint8_t> & data, size_t off) {
    T v;
    memcpy(&v, data.data() + off, sizeof(v));
    return v;
}

static std::vector<fs::path> list_files(const fs::path & dir, const std::string & suffix) {
    std::vector<fs::path> res;
    for (const auto & de : fs::directory_iterator(dir)) {
        const std::string name = de.path().filename().string();
        if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            res.push_back(de.path());
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

// a cache with the disk tier on dir and no limits; free space is mocked so the tests do not depend on the host disk
static void init_cache(server_prompt_cache & c, const fs::path & dir, const std::string & fp = FP, int32_t limit_mib = 0) {
    c.disk_free = [](const std::string &) { return (uint64_t) 1 << 60; };
    c.disk_init(dir.string(), limit_mib, fp, false);
}

// write a state to an explicit path, returns the file size (0 = failure)
static size_t write_to(const server_prompt_cache & c, const server_prompt_cache_state & st, const fs::path & path) {
    return c.disk_write_file(st, st.prompt.tokens.serialize(), path.string());
}

static server_prompt_cache::disk_entry entry_for(const fs::path & path, const llama_tokens & tokens, size_t size) {
    return server_prompt_cache::disk_entry { server_tokens(tokens, false), path.string(), size };
}

// true if the file at path is accepted either through a fresh startup scan of its directory,
// or through a stale index entry made before the file changed (tokens/size as originally written)
static bool accepted(const fs::path & path, const llama_tokens & tokens, size_t size_orig, const std::string & fp = FP) {
    server_prompt_cache c(0, 0);
    init_cache(c, path.parent_path(), fp);
    for (const auto & e : c.disk) {
        server_prompt_cache_state st;
        if (c.disk_read(e, st)) {
            return true;
        }
    }
    server_prompt_cache_state st;
    return c.disk_read(entry_for(path, tokens, size_orig), st);
}

static size_t disk_bytes(const server_prompt_cache & c) {
    size_t res = 0;
    for (const auto & e : c.disk) {
        res += e.size;
    }
    return res;
}

static size_t dir_bytes(const fs::path & dir) {
    size_t res = 0;
    for (const auto & p : list_files(dir, ".pcd")) {
        res += fs::file_size(p);
    }
    return res;
}

//
// file format
//

static void test_format(testing & t) {
    t.test("roundtrip shapes", [](testing & t) {
        tmp_dir dir("shapes");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);

        int n = 0;
        for (size_t n_tokens : {1, 3, 5000}) {
            for (int n_ckpt : {0, 1, 7}) {
                for (size_t n_main : {0, 1, 4097}) {
                    for (size_t n_drft : {0, 33}) {
                        for (size_t n_spec : {0, 5}) {
                            const auto st = make_state(make_tokens(n_tokens, 7), n_main, n_drft, n_ckpt, n_spec, (uint8_t) n);
                            const fs::path path = dir.path / ("s" + std::to_string(n++) + ".pcd");
                            const size_t size = write_to(c, st, path);
                            const std::string label = "tokens=" + std::to_string(n_tokens) + " ckpt=" + std::to_string(n_ckpt) +
                                " main=" + std::to_string(n_main) + " drft=" + std::to_string(n_drft) + " spec=" + std::to_string(n_spec);
                            t.assert_equal(label + ": returned size is the file size", (size_t) fs::file_size(path), size);

                            server_prompt_cache_state rd;
                            t.assert_true(label + ": read", c.disk_read(entry_for(path, st.prompt.tokens.get_text_tokens(), size), rd));
                            t.assert_true(label + ": byte-exact", same_state(st, rd));
                            for (const auto & ckpt : rd.prompt.checkpoints) {
                                t.assert_equal(label + ": restored checkpoint is not owned by a task", -1, ckpt->id_task);
                            }
                        }
                    }
                }
            }
        }
    });

    t.test("rewrite is byte-identical", [](testing & t) {
        tmp_dir dir("rewrite");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);

        const auto st = make_state(make_tokens(100, 3), 1000, 77, 3, 9);
        const size_t size = write_to(c, st, dir.path / "a.pcd");
        server_prompt_cache_state rd;
        t.assert_true("read", c.disk_read(entry_for(dir.path / "a.pcd", st.prompt.tokens.get_text_tokens(), size), rd));
        write_to(c, rd, dir.path / "b.pcd");
        t.assert_true("identical bytes", read_bytes(dir.path / "a.pcd") == read_bytes(dir.path / "b.pcd"));
    });

    t.test("layout", [](testing & t) {
        tmp_dir dir("layout");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);

        const auto st = make_state(make_tokens(10, 1), 100, 20, 2, 3);
        const auto packed = st.prompt.tokens.serialize();
        write_to(c, st, dir.path / "a.pcd");
        const auto data = read_bytes(dir.path / "a.pcd");

        t.assert_true("magic is the ASCII string PCD1", memcmp(data.data() + OFF_MAGIC, "PCD1", 4) == 0);
        t.assert_equal("version", (uint32_t) 2, peek<uint32_t>(data, OFF_VERSION));
        t.assert_equal("fingerprint size", (uint64_t) FP.size(), peek<uint64_t>(data, OFF_FP_LEN));
        t.assert_true("fingerprint", std::string((const char *) data.data() + OFF_FP, FP.size()) == FP);
        t.assert_equal("tokens size", (uint64_t) packed.size(), peek<uint64_t>(data, OFF_FP + FP.size()));

        size_t expected = 4 + 4 + 8 + FP.size() + 8 + packed.size() + 8 + st.data.main.size() + 8 + st.data.drft.size() + 4;
        for (const auto & ckpt : st.prompt.checkpoints) {
            expected += 8 + 4 + 4 + 8 + ckpt->data_tgt.size() + 8 + ckpt->data_dft.size() + 8 + ckpt->data_spec.size();
        }
        expected += DIGEST_SIZE;
        t.assert_equal("file size", expected, data.size());

        // the digest covers the body only: from the tokens size field up to the digest itself
        const size_t off_body = OFF_FP + FP.size();
        const uint64_t digest = XXH3_64bits(data.data() + off_body, data.size() - DIGEST_SIZE - off_body);
        t.assert_equal("digest is XXH3-64 of the body", digest, peek<uint64_t>(data, data.size() - DIGEST_SIZE));
    });

    t.test("xxh3 known answer", [](testing & t) {
        // reference values of XXH3-64 with seed 0, guards against a non-canonical build of the vendored xxhash
        t.assert_equal("empty input", (uint64_t) 0x2D06800538D394C2ull, (uint64_t) XXH3_64bits("", 0));
    });

    t.test("long fingerprint", [](testing & t) {
        // the fingerprint holds the model path, which may be longer than any fixed bound
        tmp_dir dir("longfp");
        const std::string fp = FP + std::string(5000, 'x');
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path, fp);

        const auto st = make_state(make_tokens(10, 1), 100, 0, 1, 0);
        t.assert_true("written", c.disk_write(st));
        t.assert_equal("indexed", (size_t) 1, c.disk.size());

        server_prompt_cache_state rd;
        t.assert_true("read back", c.disk_read(c.disk.front(), rd));

        server_prompt_cache c2(0, 0);
        init_cache(c2, dir.path, fp);
        t.assert_equal("found by a startup scan", (size_t) 1, c2.disk.size());
    });

    t.test("checkpoint with empty target is rejected", [](testing & t) {
        // the server never creates one; documents that the writer accepts what the reader rejects
        tmp_dir dir("emptytgt");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);

        const auto st = make_state(make_tokens(10, 1), 100, 0, 1, 0, 1, 0);
        const size_t size = write_to(c, st, dir.path / "a.pcd");
        server_prompt_cache_state rd;
        t.assert_true("rejected", !c.disk_read(entry_for(dir.path / "a.pcd", st.prompt.tokens.get_text_tokens(), size), rd));
    });
}

//
// corrupted files: every case must be rejected without a crash, an exception or a large allocation
//

static void test_corruption(testing & t) {
    // a small file with every section present
    const auto st     = make_state(make_tokens(3, 11), 5, 3, 1, 2);
    const auto tokens = st.prompt.tokens.get_text_tokens();
    const auto packed = st.prompt.tokens.serialize();
    const size_t off_tok_len = OFF_FP + FP.size();
    const size_t off_main    = off_tok_len + 8 + packed.size();
    const size_t off_drft    = off_main + 8 + st.data.main.size();
    const size_t off_n_ckpt  = off_drft + 8 + st.data.drft.size();
    const size_t off_ckpt    = off_n_ckpt + 4;
    const size_t off_tgt     = off_ckpt + 16;

    auto make_valid = [&](const fs::path & dir) {
        server_prompt_cache c(0, 0);
        init_cache(c, dir);
        write_to(c, st, dir / "a.pcd");
        return read_bytes(dir / "a.pcd");
    };

    auto expect_rejected = [&](testing & t, const std::string & msg, const fs::path & dir, const std::vector<uint8_t> & data, size_t size_orig) {
        write_bytes(dir / "a.pcd", data);
        bool ok = true;
        const size_t max_alloc = max_alloc_of([&] { ok = accepted(dir / "a.pcd", tokens, size_orig); });
        t.assert_true(msg + ": rejected", !ok);
        t.assert_true(msg + ": no large allocation (" + std::to_string(max_alloc) + " bytes)", max_alloc < ALLOC_BOUND);
        t.assert_true(msg + ": file left in place by the scan", fs::exists(dir / "a.pcd"));
    };

    t.test("valid file is accepted", [&](testing & t) {
        tmp_dir dir("valid");
        const auto good = make_valid(dir.path);
        write_bytes(dir.path / "a.pcd", good);
        t.assert_true("accepted", accepted(dir.path / "a.pcd", tokens, good.size()));
        t.assert_equal("checkpoint target offset matches the layout", (uint64_t) st.prompt.checkpoints.front()->data_tgt.size(), peek<uint64_t>(good, off_tgt));
    });

    t.test("truncated at every offset", [&](testing & t) {
        tmp_dir dir("trunc");
        const auto good = make_valid(dir.path);
        for (size_t n = 0; n < good.size(); ++n) {
            expect_rejected(t, "truncated to " + std::to_string(n), dir.path, std::vector<uint8_t>(good.begin(), good.begin() + n), good.size());
        }
    });

    t.test("trailing bytes", [&](testing & t) {
        tmp_dir dir("trailing");
        auto data = make_valid(dir.path);
        data.push_back(0);
        expect_rejected(t, "one trailing byte", dir.path, data, data.size());
        data.insert(data.end(), 4096, 0xff);
        expect_rejected(t, "many trailing bytes", dir.path, data, data.size());
    });

    t.test("header bit flips", [&](testing & t) {
        // magic, version, fingerprint size, fingerprint and token section size
        tmp_dir dir("bitflip");
        const auto good = make_valid(dir.path);
        for (size_t off = 0; off < off_tok_len + 8; ++off) {
            for (int bit = 0; bit < 8; ++bit) {
                auto data = good;
                data[off] ^= (uint8_t) (1u << bit);
                expect_rejected(t, "byte " + std::to_string(off) + " bit " + std::to_string(bit), dir.path, data, good.size());
            }
        }
    });

    t.test("token payload changed after indexing", [&](testing & t) {
        // the index holds the tokens read at startup, the file must still match them when it is loaded
        tmp_dir dir("tokchange");
        const auto good = make_valid(dir.path);
        // the last token (13) inside the packed token section, 13 -> 15 keeps the section well-formed
        const llama_token last = tokens.back();
        const auto it = std::find_end(packed.begin(), packed.end(), (const char *) &last, (const char *) &last + sizeof(last));
        t.assert_true("token found in the packed section", it != packed.end());
        auto data = good;
        data[off_tok_len + 8 + (it - packed.begin())] ^= 2;
        write_bytes(dir.path / "a.pcd", data);
        {
            server_prompt_cache c(0, 0);
            init_cache(c, dir.path);
            t.assert_equal("a fresh scan indexes the changed tokens", (size_t) 1, c.disk.size());
            t.assert_equal("changed token", (llama_token) 15, c.disk.front().tokens[2]);
            server_prompt_cache_state rd;
            t.assert_true("rejected by the digest when loaded", !c.disk_read(c.disk.front(), rd));
        }
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        server_prompt_cache_state rd;
        t.assert_true("stale index entry rejected", !c.disk_read(entry_for(dir.path / "a.pcd", tokens, good.size()), rd));
    });

    t.test("oversized size fields", [&](testing & t) {
        tmp_dir dir("oversize");
        const auto good = make_valid(dir.path);
        const uint64_t file_size = good.size();
        for (uint64_t v : {file_size + 1, (uint64_t) 1 << 32, (uint64_t) 1 << 40, UINT64_MAX - 3, UINT64_MAX}) {
            for (auto [name, off] : std::vector<std::pair<std::string, size_t>> {
                    {"fingerprint", OFF_FP_LEN}, {"tokens", off_tok_len}, {"main", off_main}, {"drft", off_drft}, {"ckpt tgt", off_tgt} }) {
                auto data = good;
                poke<uint64_t>(data, off, v);
                expect_rejected(t, name + " size " + std::to_string(v), dir.path, data, good.size());
            }
        }
        for (uint32_t v : {2u, 1000u, UINT32_MAX}) {
            auto data = good;
            poke<uint32_t>(data, off_n_ckpt, v);
            expect_rejected(t, "checkpoint count " + std::to_string(v), dir.path, data, good.size());
        }
    });

    t.test("size equal to the file size", [&](testing & t) {
        // passes the n_max bound, must then fail on the short read
        tmp_dir dir("eqsize");
        const auto good = make_valid(dir.path);
        auto data = good;
        poke<uint64_t>(data, off_main, (uint64_t) good.size());
        expect_rejected(t, "main size = file size", dir.path, data, good.size());
    });

    t.test("token size not a multiple of the token size", [&](testing & t) {
        tmp_dir dir("misaligned");
        const auto good = make_valid(dir.path);
        for (uint64_t delta : {1, 2, 3}) {
            auto data = good;
            poke<uint64_t>(data, off_tok_len, (uint64_t) packed.size() - 4 + delta);
            expect_rejected(t, "tokens size " + std::to_string(packed.size() - 4 + delta), dir.path, data, good.size());
        }
    });

    t.test("wrong magic", [&](testing & t) {
        tmp_dir dir("magic");
        const auto good = make_valid(dir.path);
        for (uint32_t v : {0u, 0x32444350u /* PCD2 */, 0x50434431u /* byte-swapped */, UINT32_MAX}) {
            auto data = good;
            poke<uint32_t>(data, OFF_MAGIC, v);
            expect_rejected(t, "magic " + std::to_string(v), dir.path, data, good.size());
        }
    });

    t.test("wrong version", [&](testing & t) {
        tmp_dir dir("version");
        const auto good = make_valid(dir.path);
        for (uint32_t v : {0u, 1u, 3u, UINT32_MAX}) {
            auto data = good;
            poke<uint32_t>(data, OFF_VERSION, v);
            expect_rejected(t, "version " + std::to_string(v), dir.path, data, good.size());
        }
    });

    t.test("wrong fingerprint", [&](testing & t) {
        tmp_dir dir("fp");
        const auto good = make_valid(dir.path);
        write_bytes(dir.path / "a.pcd", good);
        for (const std::string & fp : { std::string(), FP.substr(0, FP.size() - 1), FP + "0", "X" + FP.substr(1),
                                        FP.substr(0, FP.size() - 1) + "1" /* draft present */ }) {
            server_prompt_cache c(0, 0);
            init_cache(c, dir.path, fp);
            t.assert_equal("'" + fp + "': not indexed", (size_t) 0, c.disk.size());
            server_prompt_cache_state rd;
            t.assert_true("'" + fp + "': not readable", !c.disk_read(entry_for(dir.path / "a.pcd", tokens, good.size()), rd));
            t.assert_true("'" + fp + "': file kept for its own model", fs::exists(dir.path / "a.pcd"));
        }
    });

    t.test("bit flip anywhere in the body", [&](testing & t) {
        // every byte after the header, the digest included; a flip there must never load
        tmp_dir dir("bodyflip");
        const auto good = make_valid(dir.path);
        const size_t off_dft  = off_tgt + 8 + st.prompt.checkpoints.front()->data_tgt.size();
        const size_t off_spec = off_dft + 8 + st.prompt.checkpoints.front()->data_dft.size();
        const size_t off_dig  = good.size() - DIGEST_SIZE;
        auto region = [&](size_t off) -> std::string {
            if (off <  off_main)   return "tokens";
            if (off <  off_drft)   return "main";
            if (off <  off_n_ckpt) return "drft";
            if (off <  off_ckpt)   return "n_ckpt";
            if (off <  off_tgt)    return "ckpt n_tokens/pos";
            if (off <  off_dft)    return "ckpt tgt";
            if (off <  off_spec)   return "ckpt dft";
            if (off <  off_dig)    return "ckpt spec";
            return "digest";
        };
        std::set<std::string> covered;
        for (size_t off = off_tok_len; off < good.size(); ++off) {
            // every bit of the digest and of one payload byte per region, one bit elsewhere
            const bool all_bits = off >= off_dig || off == off_main + 8 || off == off_drft + 8 || off == off_tgt + 8;
            for (int bit = all_bits ? 0 : (int) (off % 8); bit < (all_bits ? 8 : (int) (off % 8) + 1); ++bit) {
                auto data = good;
                data[off] ^= (uint8_t) (1u << bit);
                expect_rejected(t, region(off) + " byte " + std::to_string(off) + " bit " + std::to_string(bit), dir.path, data, good.size());
            }
            covered.insert(region(off));
        }
        t.assert_equal("all regions covered", (size_t) 9, covered.size());
    });

    t.test("version 1 file", [&](testing & t) {
        // the format before the digest: same layout, version 1, no digest at the end
        tmp_dir dir("v1");
        auto data = make_valid(dir.path);
        poke<uint32_t>(data, OFF_VERSION, 1);
        data.resize(data.size() - DIGEST_SIZE);
        expect_rejected(t, "version 1", dir.path, data, data.size());

        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        t.assert_equal("not indexed", (size_t) 0, c.disk.size());
        t.assert_true("not deleted by the scan, like files of another model", fs::exists(dir.path / "a.pcd"));
    });
}

//
// startup scan
//

static void test_init(testing & t) {
    t.test("creates a missing directory", [](testing & t) {
        tmp_dir dir("mkdir");
        const fs::path sub = dir.path / "a" / "b" / "c";
        server_prompt_cache c(0, 0);
        init_cache(c, sub);
        t.assert_true("created", fs::is_directory(sub));
        t.assert_equal("enabled", sub.string(), c.disk_dir);
        t.assert_true("writes", c.disk_write(make_state(make_tokens(5, 1), 10, 0, 0, 0)));
    });

    t.test("uncreatable directory disables the tier", [](testing & t) {
        tmp_dir dir("nodir");
        write_bytes(dir.path / "file", {1, 2, 3});
        for (const fs::path & bad : { dir.path / "file" / "sub", dir.path / "file" }) {
            server_prompt_cache c(0, 0);
            init_cache(c, bad);
            t.assert_true(bad.string() + ": disabled", c.disk_dir.empty());

            // eviction still drops the RAM entry, and writes nothing
            c.states.push_back(make_state(make_tokens(5, 1), 10, 0, 0, 0));
            c.evict_front();
            t.assert_true(bad.string() + ": evicted", c.states.empty());
            t.assert_equal(bad.string() + ": not written", (size_t) 0, c.disk.size());
        }
        t.assert_true("file untouched", read_bytes(dir.path / "file") == std::vector<uint8_t>({1, 2, 3}));
    });

    t.test("cleans up only its own temporary files", [](testing & t) {
        tmp_dir dir("tmpfiles");
        write_bytes(dir.path / "0000000000000001-0.pcd.tmp", {1});
        write_bytes(dir.path / "other.tmp", {2});              // not ours, e.g. --cache-disk /tmp
        write_bytes(dir.path / "notes.txt", {3});
        write_bytes(dir.path / "garbage.pcd", {4, 5, 6});      // invalid, ignored and kept
        fs::create_directories(dir.path / "subdir.pcd");       // a directory with our extension

        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        t.assert_true(".pcd.tmp removed", !fs::exists(dir.path / "0000000000000001-0.pcd.tmp"));
        t.assert_true("foreign .tmp kept", fs::exists(dir.path / "other.tmp"));
        t.assert_true("other file kept", fs::exists(dir.path / "notes.txt"));
        t.assert_true("invalid .pcd kept", fs::exists(dir.path / "garbage.pcd"));
        t.assert_true("directory kept", fs::is_directory(dir.path / "subdir.pcd"));
        t.assert_equal("nothing indexed", (size_t) 0, c.disk.size());
    });

    t.test("orders by mtime", [](testing & t) {
        tmp_dir dir("mtime");
        server_prompt_cache w(0, 0);
        init_cache(w, dir.path);
        // names sort a < b < c, mtimes sort b < c < a
        const auto now = fs::file_time_type::clock::now();
        const std::vector<std::pair<std::string, int>> files = { {"a.pcd", 0}, {"b.pcd", -20}, {"c.pcd", -10} };
        int i = 0;
        for (const auto & [name, age] : files) {
            write_to(w, make_state(make_tokens(5, 100*(++i)), 10, 0, 0, 0), dir.path / name);
            fs::last_write_time(dir.path / name, now + std::chrono::seconds(age));
        }
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        std::vector<std::string> order;
        for (const auto & e : c.disk) {
            order.push_back(fs::path(e.path).filename().string());
        }
        t.assert_true("oldest first", order == std::vector<std::string>({"b.pcd", "c.pcd", "a.pcd"}));
        t.assert_equal("first entry tokens", (llama_token) 200, c.disk.front().tokens[0]);
        t.assert_equal("entry size is the file size", (size_t) fs::file_size(dir.path / "b.pcd"), c.disk.front().size);
    });

    t.test("trims to the limit at startup", [](testing & t) {
        tmp_dir dir("inittrim");
        server_prompt_cache w(0, 0);
        init_cache(w, dir.path);
        const auto now = fs::file_time_type::clock::now();
        const size_t n_main = 600*1024; // two of these do not fit in 1 MiB
        for (int i = 0; i < 3; ++i) {
            const fs::path p = dir.path / ("f" + std::to_string(i) + ".pcd");
            write_to(w, make_state(make_tokens(5, 100*(i + 1)), n_main, 0, 0, 0), p);
            fs::last_write_time(p, now + std::chrono::seconds(i));
        }
        write_bytes(dir.path / "foreign.pcd", std::vector<uint8_t>(2*1024*1024, 0)); // not counted, not deleted

        server_prompt_cache c(0, 0);
        init_cache(c, dir.path, FP, 1);
        t.assert_equal("one entry left", (size_t) 1, c.disk.size());
        t.assert_true("newest kept", fs::exists(dir.path / "f2.pcd"));
        t.assert_true("oldest deleted", !fs::exists(dir.path / "f0.pcd"));
        t.assert_true("second oldest deleted", !fs::exists(dir.path / "f1.pcd"));
        t.assert_true("foreign file kept", fs::exists(dir.path / "foreign.pcd"));
        t.assert_true("within limit", disk_bytes(c) <= 1024*1024);
    });

    t.test("limit 0 keeps everything", [](testing & t) {
        tmp_dir dir("nolimit");
        server_prompt_cache w(0, 0);
        init_cache(w, dir.path);
        for (int i = 0; i < 3; ++i) {
            t.assert_true("written", w.disk_write(make_state(make_tokens(5, 100*(i + 1)), 600*1024, 0, 0, 0)));
        }
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path, FP, 0);
        t.assert_equal("all indexed", (size_t) 3, c.disk.size());
        t.assert_equal("limit", (size_t) 0, c.disk_limit);
    });
}

//
// size limit
//

static void test_trim(testing & t) {
    // three real files, the sizes in the index are set by hand to make the boundaries exact
    auto setup = [](server_prompt_cache & c, const fs::path & dir) {
        init_cache(c, dir);
        for (int i = 0; i < 3; ++i) {
            c.disk_write(make_state(make_tokens(5, 100*(i + 1)), 10, 0, 0, 0));
        }
        for (auto & e : c.disk) {
            e.size = 1000;
        }
    };

    t.test("total exactly at the limit", [&](testing & t) {
        tmp_dir dir("trim-eq");
        server_prompt_cache c(0, 0);
        setup(c, dir.path);
        c.disk_limit = 3000;
        c.disk_trim(0);
        t.assert_equal("nothing deleted", (size_t) 3, c.disk.size());
        c.disk_limit = 3500;
        c.disk_trim(500);
        t.assert_equal("total + new exactly at the limit", (size_t) 3, c.disk.size());
        t.assert_equal("files", (size_t) 3, list_files(dir.path, ".pcd").size());
    });

    t.test("one byte over", [&](testing & t) {
        tmp_dir dir("trim-over");
        server_prompt_cache c(0, 0);
        setup(c, dir.path);
        const std::string oldest = c.disk.front().path;
        c.disk_limit = 2999;
        c.disk_trim(0);
        t.assert_equal("one deleted", (size_t) 2, c.disk.size());
        t.assert_true("oldest deleted", !fs::exists(oldest));
        t.assert_equal("files", (size_t) 2, list_files(dir.path, ".pcd").size());
    });

    t.test("one byte over with a new entry", [&](testing & t) {
        tmp_dir dir("trim-new");
        server_prompt_cache c(0, 0);
        setup(c, dir.path);
        c.disk_limit = 3500;
        c.disk_trim(501);
        t.assert_equal("one deleted", (size_t) 2, c.disk.size());
    });

    t.test("limit 0 means no limit", [&](testing & t) {
        tmp_dir dir("trim-zero");
        server_prompt_cache c(0, 0);
        setup(c, dir.path);
        c.disk_limit = 0;
        c.disk_trim(SIZE_MAX / 2);
        t.assert_equal("nothing deleted", (size_t) 3, c.disk.size());
    });

    t.test("entry bigger than the whole limit", [&](testing & t) {
        tmp_dir dir("trim-big");
        server_prompt_cache c(0, 0);
        setup(c, dir.path);
        c.disk_limit = 4096;
        t.assert_true("not written", !c.disk_write(make_state(make_tokens(5, 900), 8192, 0, 0, 0)));
        t.assert_equal("existing entries kept", (size_t) 3, c.disk.size());
        t.assert_equal("existing files kept", (size_t) 3, list_files(dir.path, ".pcd").size());
    });

    t.test("write evicts oldest first", [](testing & t) {
        tmp_dir dir("trim-lru");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        std::vector<std::string> paths;
        for (int i = 0; i < 4; ++i) {
            t.assert_true("written", c.disk_write(make_state(make_tokens(5, 100*(i + 1)), 4000, 0, 0, 0)));
            paths.push_back(c.disk.back().path);
            if (i == 2) {
                // room for the three written so far plus a little, the fourth pushes the first out
                c.disk_limit = disk_bytes(c) + 2048;
            }
        }
        t.assert_true("first deleted", !fs::exists(paths[0]));
        for (int i = 1; i < 4; ++i) {
            t.assert_true("newer kept " + std::to_string(i), fs::exists(paths[i]));
        }
        t.assert_true("within limit", disk_bytes(c) <= c.disk_limit);
    });

    t.test("limit holds with many checkpoints", [](testing & t) {
        // the size estimate used before writing must cover the per-checkpoint headers
        tmp_dir dir("trim-ckpt");
        const auto e1 = make_state(make_tokens(50, 1000), 2000, 0, 0, 0);
        const auto e2 = make_state(make_tokens(50, 5000), 2000, 0, 64, 0);

        size_t s1 = 0;
        size_t s2 = 0;
        {
            server_prompt_cache c(0, 0);
            init_cache(c, dir.path);
            s1 = write_to(c, e1, dir.path / "x1");
            s2 = write_to(c, e2, dir.path / "x2");
            fs::remove(dir.path / "x1");
            fs::remove(dir.path / "x2");
        }

        int n_bad = 0;
        for (size_t limit = s1 + s2 - 3000; limit <= s1 + s2 + 3000; limit += 7) {
            server_prompt_cache c(0, 0);
            init_cache(c, dir.path);
            c.disk_limit = limit;
            c.disk_write(e1);
            c.disk_write(e2);
            if (disk_bytes(c) > limit || dir_bytes(dir.path) > limit) {
                n_bad++;
            }
            for (const auto & p : list_files(dir.path, ".pcd")) {
                fs::remove(p);
            }
        }
        t.assert_equal("limits exceeded", 0, n_bad);
    });
}

//
// writing
//

static void test_write(testing & t) {
    t.test("superset already on disk", [](testing & t) {
        tmp_dir dir("dedupe");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        t.assert_true("long entry written", c.disk_write(make_state(make_tokens(10, 1), 100, 0, 0, 0)));
        t.assert_true("same entry not written again", !c.disk_write(make_state(make_tokens(10, 1), 100, 0, 0, 0)));
        t.assert_equal("one file", (size_t) 1, list_files(dir.path, ".pcd").size());
        t.assert_true("longer entry written", c.disk_write(make_state(make_tokens(12, 1), 100, 0, 0, 0)));
        t.assert_equal("two files, the shorter one is not removed", (size_t) 2, list_files(dir.path, ".pcd").size());
    });

    // known limitation: a file covering a longer prompt does not help a model that cannot roll back
    // (recurrent/hybrid/SWA) unless it carries a checkpoint before the divergence point; the shorter
    // entry is still dropped, and that prompt has to be processed again
    t.test("known limitation: shorter prefix not written when a longer file exists", [](testing & t) {
        tmp_dir dir("prefix");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        t.assert_true("long entry written", c.disk_write(make_state(make_tokens(10, 1), 100, 0, 0, 0)));
        t.assert_true("prefix entry skipped", !c.disk_write(make_state(make_tokens(5, 1), 100, 0, 1, 0)));
        t.assert_equal("one file", (size_t) 1, list_files(dir.path, ".pcd").size());
    });

    t.test("unique file names", [](testing & t) {
        tmp_dir dir("names");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        const int n = 64;
        std::set<std::string> names;
        const std::regex re("[0-9a-f]{16}-[0-9]+\\.pcd");
        for (int i = 0; i < n; ++i) {
            t.assert_true("written", c.disk_write(make_state(make_tokens(3, 10*(i + 1)), 8, 0, 0, 0)));
            const std::string name = fs::path(c.disk.back().path).filename().string();
            t.assert_true("name format: " + name, std::regex_match(name, re));
            t.assert_true("sequence suffix", name.size() > 4 && name.find("-" + std::to_string(i) + ".pcd") != std::string::npos);
            names.insert(name);
        }
        t.assert_equal("distinct names", (size_t) n, names.size());
        t.assert_equal("files", (size_t) n, list_files(dir.path, ".pcd").size());
        t.assert_equal("sequence", (uint64_t) n, c.disk_seq);
        t.assert_equal("no temporary files", (size_t) 0, list_files(dir.path, ".tmp").size());
    });

    t.test("free space headroom", [](testing & t) {
        tmp_dir dir("space");
        const auto st = make_state(make_tokens(100, 1), 4096, 0, 2, 0);

        auto try_avail = [&](uint64_t avail) {
            server_prompt_cache c(0, 0);
            init_cache(c, dir.path);
            c.disk_free = [avail](const std::string &) { return avail; };
            const bool ok = c.disk_write(st);
            t.assert_equal("index matches the result", ok ? (size_t) 1 : (size_t) 0, c.disk.size());
            for (const auto & p : list_files(dir.path, ".pcd")) {
                fs::remove(p);
            }
            return ok;
        };

        const uint64_t head = server_prompt_cache::disk_headroom;
        t.assert_true("query failed (0 free)", !try_avail(0));

        size_t size_file = 0;
        {
            server_prompt_cache c(0, 0);
            init_cache(c, dir.path);
            size_file = write_to(c, st, dir.path / "probe");
            fs::remove(dir.path / "probe");
        }

        t.assert_true("one byte short of file + headroom", !try_avail(head + size_file - 1));
        t.assert_true("plenty", try_avail(head + size_file + 1024*1024));

        // smallest accepted value: at least file + headroom, and the estimate is not far above the real size
        uint64_t lo = head + size_file - 1;          // rejected
        uint64_t hi = head + size_file + 1024*1024;  // accepted
        while (hi - lo > 1) {
            const uint64_t mid = lo + (hi - lo)/2;
            (try_avail(mid) ? hi : lo) = mid;
        }
        t.assert_true("threshold covers the file and the headroom", hi >= head + size_file);
        t.assert_true("threshold is tight (" + std::to_string(hi - head - size_file) + " bytes over)", hi - head - size_file <= 8192);
        t.assert_true("rejected just below the threshold", !try_avail(hi - 1));
        t.assert_true("accepted at the threshold", try_avail(hi));
    });

    t.test("rename failure leaves nothing behind", [](testing & t) {
        tmp_dir dir("rename");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        // a non-empty directory cannot be replaced by a file
        fs::create_directories(dir.path / "x.pcd" / "keep");
        const auto st = make_state(make_tokens(5, 1), 100, 0, 1, 0);
        t.assert_equal("failed", (size_t) 0, write_to(c, st, dir.path / "x.pcd"));
        t.assert_true("no temporary file", !fs::exists(dir.path / "x.pcd.tmp"));
        t.assert_true("directory untouched", fs::is_directory(dir.path / "x.pcd" / "keep"));
    });

    t.test("directory removed after startup", [](testing & t) {
        tmp_dir dir("gone");
        const fs::path sub = dir.path / "sub";
        server_prompt_cache c(0, 0);
        init_cache(c, sub);
        fs::remove_all(sub);
        t.assert_true("not written", !c.disk_write(make_state(make_tokens(5, 1), 100, 0, 0, 0)));
        t.assert_equal("no index entry", (size_t) 0, c.disk.size());
        t.assert_true("not recreated with junk", !fs::exists(sub) || list_files(sub, "").empty());
    });

#ifndef _WIN32
    t.test("write failure leaves nothing behind", [](testing & t) {
        // RLIMIT_FSIZE makes the write fail half way, like a full disk (also as root)
        tmp_dir dir("wfail");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        t.assert_true("first entry written", c.disk_write(make_state(make_tokens(5, 1), 100, 0, 0, 0)));

        struct rlimit old_lim;
        getrlimit(RLIMIT_FSIZE, &old_lim);
        auto old_sig = signal(SIGXFSZ, SIG_IGN);
        struct rlimit lim = old_lim;
        lim.rlim_cur = 64*1024;
        setrlimit(RLIMIT_FSIZE, &lim);

        const bool ok = c.disk_write(make_state(make_tokens(5, 100), 1024*1024, 0, 1, 0));

        setrlimit(RLIMIT_FSIZE, &old_lim);
        signal(SIGXFSZ, old_sig);

        t.assert_true("failed", !ok);
        t.assert_equal("index unchanged", (size_t) 1, c.disk.size());
        t.assert_equal("one file", (size_t) 1, list_files(dir.path, ".pcd").size());
        t.assert_equal("no temporary file", (size_t) 0, list_files(dir.path, ".tmp").size());
    });
#endif
}

//
// eviction from RAM goes through the disk tier
//

static void test_evict(testing & t) {
    t.test("alloc spills the oldest entry", [](testing & t) {
        tmp_dir dir("alloc");
        server_prompt_cache c(1, 0); // 1 MiB
        init_cache(c, dir.path);
        c.states.push_back(make_state(make_tokens(10, 1), 600*1024, 0, 1, 0));

        server_prompt prompt { server_tokens(make_tokens(10, 500), false), {} };
        t.assert_true("allocated", c.alloc(prompt, 600*1024, 0) != nullptr);
        t.assert_equal("one RAM entry", (size_t) 1, c.states.size());
        t.assert_equal("one disk entry", (size_t) 1, c.disk.size());
        t.assert_equal("the evicted prompt", (llama_token) 1, c.disk.front().tokens[0]);

        server_prompt_cache_state rd;
        t.assert_true("readable", c.disk_read(c.disk.front(), rd));
        t.assert_equal("with its checkpoint", (size_t) 1, rd.prompt.checkpoints.size());
    });

    t.test("update spills on the size limit", [](testing & t) {
        tmp_dir dir("update");
        server_prompt_cache c(1, 0);
        init_cache(c, dir.path);
        c.states.push_back(make_state(make_tokens(10, 1),   600*1024, 0, 0, 0));
        c.states.push_back(make_state(make_tokens(10, 100), 600*1024, 0, 0, 0));
        c.update();
        t.assert_equal("one RAM entry", (size_t) 1, c.states.size());
        t.assert_equal("one disk entry", (size_t) 1, c.disk.size());
        t.assert_equal("oldest spilled", (llama_token) 1, c.disk.front().tokens[0]);
    });

    t.test("update spills on the token limit", [](testing & t) {
        tmp_dir dir("update-tok");
        server_prompt_cache c(0, 15);
        init_cache(c, dir.path);
        c.states.push_back(make_state(make_tokens(10, 1),   10, 0, 0, 0));
        c.states.push_back(make_state(make_tokens(10, 100), 10, 0, 0, 0));
        c.update();
        t.assert_equal("one RAM entry", (size_t) 1, c.states.size());
        t.assert_equal("one disk entry", (size_t) 1, c.disk.size());
    });

    t.test("flush writes the newest entries that fit, oldest first", [](testing & t) {
        tmp_dir dir("flush");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path, FP, 1); // 1 MiB cap: two of the three 400 KiB entries fit
        c.states.push_back(make_state(make_tokens(10, 1),   400*1024, 0, 0, 0));
        c.states.push_back(make_state(make_tokens(10, 100), 400*1024, 0, 0, 0));
        c.states.push_back(make_state(make_tokens(10, 200), 400*1024, 0, 0, 0));
        c.disk_flush();
        t.assert_equal("two disk entries", (size_t) 2, c.disk.size());
        t.assert_equal("older of the two first", (llama_token) 100, c.disk.front().tokens[0]);
        t.assert_equal("newest last",            (llama_token) 200, c.disk.back().tokens[0]);
        t.assert_equal("RAM untouched", (size_t) 3, c.states.size());

        server_prompt_cache r(0, 0);
        init_cache(r, dir.path, FP, 1);
        t.assert_equal("reloaded", (size_t) 2, r.disk.size());
        t.assert_equal("reload keeps the order", (llama_token) 200, r.disk.back().tokens[0]);
    });

    t.test("flush skips entries already on disk", [](testing & t) {
        tmp_dir dir("flush-dup");
        server_prompt_cache c(0, 0);
        init_cache(c, dir.path);
        c.states.push_back(make_state(make_tokens(10, 1), 100, 0, 0, 0));
        c.disk_flush();
        c.disk_flush();
        t.assert_equal("one disk entry", (size_t) 1, c.disk.size());
    });

    t.test("disabled tier only drops", [](testing & t) {
        server_prompt_cache c(1, 0);
        c.states.push_back(make_state(make_tokens(10, 1),   600*1024, 0, 0, 0));
        c.states.push_back(make_state(make_tokens(10, 100), 600*1024, 0, 0, 0));
        c.update();
        t.assert_equal("one RAM entry", (size_t) 1, c.states.size());
        t.assert_equal("no disk entry", (size_t) 0, c.disk.size());
    });
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    if (!PCD_TRACK_ALLOC) {
        t.out << "note: allocation tracking is off under a sanitizer, allocation bounds are not checked\n";
    }

    t.test("format",     test_format);
    t.test("corruption", test_corruption);
    t.test("init",       test_init);
    t.test("trim",       test_trim);
    t.test("write",      test_write);
    t.test("evict",      test_evict);

    return t.summary();
}

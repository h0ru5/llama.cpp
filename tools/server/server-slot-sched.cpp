#include "server-slot-sched.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>

#define XXH_STATIC_LINKING_ONLY // XXH3_state_t on the stack
#include "hash/xxhash/xxhash.h"

//
// server_prompt
//

server_prompt_reuse server_prompt::get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const {
    server_prompt_reuse res;
    res.n_past   = n_prefix;
    res.pos_next = tokens.pos_next(n_prefix);

    // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
    // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
    const llama_pos pos_eval = res.pos_next - (has_new_tokens ? 0 : 1);

    // a state or a checkpoint is useful when its pos_min is below this threshold
    const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

    if (pos_min < pos_min_thold) {
        return res;
    }

    // search for a context checkpoint
    const auto it = std::find_if(
        checkpoints.rbegin(),
        checkpoints.rend(),
        [&](const auto & ckpt) {
            const auto & cur = *ckpt;
            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
            SRV_TRC("checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
            if (cur.pos_max > res.pos_next) {
                return false;
            }
            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
        }
    );

    if (it == checkpoints.rend()) {
        res.n_past   = 0;
        res.pos_next = 0;
        return res;
    }

    const auto & cur = **it;

    res.pos_next = std::min(res.pos_next, std::max(cur.pos_min + 1, cur.pos_max));
    res.n_past   = std::min(tokens.size_up_to_pos(res.pos_next), (size_t) cur.n_tokens);
    res.ckpt     = *it;

    return res;
}

//
// server_prompt_cache
//

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size() && are_lora_equal(it->prompt.lora, prompt.lora)) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt->size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size() && are_lora_equal(it->prompt.lora, prompt.lora)) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            evict_front();
        }
    }

    common_state_data state_data_tgt;
    common_state_data state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ prompt.checkpoints,
            /*.lora        =*/ prompt.lora,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    // a prompt made with another lora config is cleared before the new one starts, so it counts as empty
    const bool is_empty = prompt.tokens.empty() || lora_should_clear_cache(prompt.lora, lora_new);

    int lcp_best = is_empty ? 0 : prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = is_empty ? -1.0f : float(lcp_best) / prompt.tokens.size(); // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (lora_should_clear_cache(it->prompt.lora, lora_new)) {
            SRV_TRC("   - prompt with length %7zu, skipping, other lora config\n", it->prompt.tokens.size());
            continue;
        }

        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
            lcp_best    = lcp_cur;

            it_best = it;
        }
    }

    // a disk entry is used only if it beats the slot and every RAM entry
    auto it_disk = disk.end();
    const bool disk_lora_ok = !lora_should_clear_cache(disk_lora, lora_new);
    const int lcp_ram = lcp_best;
    for (auto it = disk.begin(); disk_lora_ok && it != disk.end(); ++it) {
        const int lcp_cur = it->tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        if (f_keep_cur >= 0.25f && f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
            lcp_best    = lcp_cur;

            it_disk = it;
        }
    }

    // taken out of the index while it is restored, put back as most recent on success
    std::optional<disk_entry> from_disk;
    if (it_disk != disk.end()) {
        from_disk = std::move(*it_disk);
        disk.erase(it_disk);

        const int64_t t_start = ggml_time_us();
        server_prompt_cache_state state;
        if (disk_read(*from_disk, state)) {
            SRV_INF(" - read prompt with %zu tokens (%.3f MiB) from '%s' in %.2f ms\n", state.prompt.tokens.size(),
                    state.size() / (1024.0 * 1024.0), from_disk->path.c_str(), (ggml_time_us() - t_start) / 1000.0);
            states.push_back(std::move(state));
            it_best = std::prev(states.end());
        } else {
            SRV_WRN(" - failed to read prompt cache file '%s', deleting it\n", from_disk->path.c_str());
            std::error_code ec;
            std::filesystem::remove(from_disk->path, ec);
            from_disk.reset();
            lcp_best = lcp_ram;
        }
    }

    // a disk entry that does not restore (e.g. other context params) would fail on every hit - delete it,
    // and drop its RAM copy too so a later eviction does not write it back
    auto drop_disk = [&]() {
        if (from_disk) {
            std::error_code ec;
            std::filesystem::remove(from_disk->path, ec);
            states.erase(it_best);
        }
    };

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        const int64_t t_start = ggml_time_us();

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);
                drop_disk();

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);
                    drop_disk();

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_RESTORE__\n");

        // the whole cached prompt is a prefix of the new one
        if (lcp_best == (int) it_best->prompt.tokens.size()) {
            SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_FULL_RESTORE__\n");
        }

        prompt = std::move(it_best->prompt);

        states.erase(it_best);

        SRV_TRC("slot %d: prompt cache restore took %.2f ms\n", id_slot, (ggml_time_us() - t_start) / 1000.0);
    }

    if (from_disk) {
        // the file stays valid: keep it as the most recent disk entry, so it survives a server restart
        std::error_code ec;
        std::filesystem::last_write_time(from_disk->path, std::filesystem::file_time_type::clock::now(), ec);
        disk.push_back(std::move(*from_disk));
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            evict_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            evict_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

//
// server_slot_sched
//

static std::vector<const server_slot_info *> get_free_slot_infos(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) {
    std::vector<const server_slot_info *> free_slots;
    for (const auto & slot : slots) {
        if (!slot.is_processing && slot.id != exclude_id_slot) {
            free_slots.push_back(&slot);
        }
    }

    std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot_info * a, const server_slot_info * b) {
        if ((a->prompt->n_tokens() == 0) != (b->prompt->n_tokens() == 0)) {
            return a->prompt->n_tokens() == 0;
        }
        return a->t_last_used < b->t_last_used;
    });

    free_slots.resize(std::min(free_slots.size(), n_slots_needed));

    return free_slots;
}

std::vector<int> server_slot_sched::get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const {
    std::vector<int> res;
    for (const auto * slot : get_free_slot_infos(slots, n_slots_needed, exclude_id_slot)) {
        res.push_back(slot->id);
    }

    return res;
}

server_slot_plan server_slot_sched::select(const std::vector<server_slot_info> & slots, const server_task & task, const std::vector<common_adapter_lora_info> & lora) const {
    const server_slot_info * ret = nullptr;

    bool update_cache = false;

    // if a specific slot is requested, use it (still goes through cache update logic below)
    if (task.id_slot != -1 && !slots.empty()) {
        // note: allow id_slot to be out of bounds (wrap around)
        const int id_slot = task.id_slot % slots.size();

        for (const auto & slot : slots) {
            if (slot.id == id_slot) {
                ret = &slot;
                break;
            }
        }

        if (ret) {
            // a busy slot is returned untouched, the caller defers the task
            if (ret->is_processing) {
                return { ret->id, false };
            }

            SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
        }
    }

    // find the slot that has at least n% prompt similarity
    if (slot_prompt_similarity != 0.0f) {
        float f_sim_best = 0;

        for (const auto & slot : slots) {
            if (task.id_slot != -1 && slot.id != task.id_slot) {
                continue;
            }

            // skip the slot if it is not available
            if (slot.is_processing) {
                SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing);
                continue;
            }

            const auto & tokens = slot.prompt->tokens;

            // skip the slot if it does not contains cached tokens
            if (tokens.empty()) {
                SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                continue;
            }

            // the task would clear this slot, its state cannot be reused with another lora config
            if (lora_should_clear_cache(slot.prompt->lora, lora)) {
                SLT_TRC(slot, "%s", " - skipping, other lora config\n");
                continue;
            }

            // fraction of the Longest Common Prefix length with respect to the input prompt length
            const size_t lcp_len = tokens.get_common_prefix(task.tokens);
            const float f_sim_cur = float(lcp_len) / task.tokens.size();

            SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

            // select the current slot if the criteria match
            if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                f_sim_best = f_sim_cur;

                ret = &slot;
            }
        }

        if (ret != nullptr) {
            const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt->tokens.size();

            if (task.id_slot == -1) {
                SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        f_sim_best, slot_prompt_similarity, f_keep);
            }

            // if we are about to lose a large portion of the existing context - save it in the prompt cache
            if (f_keep < 0.5f) {
                update_cache = true;
            }
        }
    }

    // find an empty slot, or the one that has been least recently used
    if (ret == nullptr) {
        const auto free_slots = get_free_slot_infos(slots, 1, -1);

        if (!free_slots.empty()) {
            ret = free_slots[0];

            SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt->n_tokens(), ret->t_last_used);

            update_cache = true;
        }
    }

    if (ret == nullptr) {
        return {};
    }

    // cache prompts only for the tasks that reuse them
    update_cache = update_cache && !task.is_stateless();

    return { ret->id, update_cache };
}

//
// server_prompt_cache - disk tier
//
// file (all integers little-endian as in memory):
//   header: magic u32, version u32, model fingerprint (u64 size + bytes)
//   body:   packed tokens, target state, draft state (each u64 size + bytes), n_ckpt u32,
//           per checkpoint: n_tokens i64, pos_min i32, pos_max i32, tgt, dft, spec (each u64 size + bytes)
//   digest: XXH3-64 (seed 0) of the body, u64, last field of the file
// the header is not hashed, it is compared field by field
// written to "<name>.tmp" and renamed, so a crash or a full disk never leaves a partial ".pcd" file
//

static constexpr uint32_t PCD_MAGIC   = 0x31444350; // "PCD1"
static constexpr uint32_t PCD_VERSION = 2;          // 2: XXH3-64 digest of the body at the end

// hash in chunks while the data is still in cache, a state blob can be ~1 GiB
static constexpr size_t PCD_CHUNK = 1024*1024;

struct pcd_hash {
    XXH3_state_t st;
    bool on = false; // off for the header

    pcd_hash() {
        XXH3_INITSTATE(&st);
        XXH3_64bits_reset(&st);
    }

    void update(const void * data, size_t n) {
        if (on) {
            XXH3_64bits_update(&st, data, n);
        }
    }

    uint64_t digest() const {
        return XXH3_64bits_digest(&st);
    }
};

static void pcd_write(std::ofstream & ofs, pcd_hash & h, const void * data, size_t n) {
    const char * p = (const char *) data;
    while (n > 0) {
        const size_t k = std::min(n, PCD_CHUNK);
        h.update(p, k);
        ofs.write(p, k);
        p += k;
        n -= k;
    }
}

template <typename T>
static void pcd_write_pod(std::ofstream & ofs, pcd_hash & h, const T & v) {
    pcd_write(ofs, h, &v, sizeof(v));
}

template <typename Buf>
static void pcd_write_buf(std::ofstream & ofs, pcd_hash & h, const Buf & buf) {
    const uint64_t n = buf.size() * sizeof(buf[0]);
    pcd_write_pod(ofs, h, n);
    pcd_write(ofs, h, buf.data(), n);
}

static bool pcd_read(std::ifstream & ifs, pcd_hash & h, void * data, size_t n) {
    char * p = (char *) data;
    while (n > 0) {
        const size_t k = std::min(n, PCD_CHUNK);
        if (!ifs.read(p, k)) {
            return false;
        }
        h.update(p, k);
        p += k;
        n -= k;
    }
    return true;
}

template <typename T>
static bool pcd_read_pod(std::ifstream & ifs, pcd_hash & h, T & v) {
    return pcd_read(ifs, h, &v, sizeof(v));
}

// n_max bounds the allocation, the size field may be corrupted
template <typename Buf>
static bool pcd_read_buf(std::ifstream & ifs, pcd_hash & h, Buf & buf, uint64_t n_max) {
    using T = typename Buf::value_type;
    uint64_t n = 0;
    if (!pcd_read_pod(ifs, h, n) || n > n_max || n % sizeof(T) != 0) {
        return false;
    }
    buf.resize(n / sizeof(T));
    return pcd_read(ifs, h, buf.data(), n);
}

// returns false if the file is not an entry written for this model; the tokens are not verified yet
static bool pcd_read_head(std::ifstream & ifs, pcd_hash & h, const std::string & fp, llama_tokens & packed, uint64_t n_max) {
    uint32_t    magic   = 0;
    uint32_t    version = 0;
    std::string fp_file;
    if (!(pcd_read_pod(ifs, h, magic)   && magic   == PCD_MAGIC &&
          pcd_read_pod(ifs, h, version) && version == PCD_VERSION &&
          pcd_read_buf(ifs, h, fp_file, fp.size()) && fp_file == fp)) {
        return false;
    }
    h.on = true;
    return pcd_read_buf(ifs, h, packed, n_max);
}

void server_prompt_cache::disk_init(const std::string & dir, int32_t limit_mib, const std::string & fp, bool has_mtmd) {
    namespace fs = std::filesystem;

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        SRV_ERR("failed to create prompt cache directory '%s': %s - disk tier disabled\n", dir.c_str(), ec.message().c_str());
        return;
    }

    disk_dir   = dir;
    disk_fp    = fp;
    disk_limit = 1024ull*1024ull*limit_mib;
    disk_mtmd  = has_mtmd;

    std::vector<std::pair<fs::file_time_type, disk_entry>> found;
    for (const auto & de : fs::directory_iterator(dir, ec)) {
        const fs::path & path = de.path();
        std::error_code ec2;
        if (path.extension() == ".tmp" && path.stem().extension() == ".pcd") {
            fs::remove(path, ec2); // left over from an interrupted write, other .tmp files are not ours
            continue;
        }
        if (path.extension() != ".pcd") {
            continue;
        }
        const uint64_t size = de.file_size(ec2);
        std::ifstream ifs(path, std::ios::binary);
        llama_tokens packed;
        pcd_hash h; // the startup scan reads only the head, the digest is checked when the entry is loaded
        if (ec2 || !pcd_read_head(ifs, h, fp, packed, size)) {
            SRV_WRN("ignoring prompt cache file '%s' (another model or invalid)\n", path.c_str());
            continue;
        }
        try {
            found.emplace_back(de.last_write_time(ec2), disk_entry { server_tokens::deserialize(packed, has_mtmd), path.string(), size });
        } catch (const std::exception & e) {
            SRV_WRN("ignoring prompt cache file '%s': %s\n", path.c_str(), e.what());
        }
    }

    // mtime ticks are coarse, files written in one tick keep their write order through the name (time-seq)
    std::sort(found.begin(), found.end(), [](const auto & a, const auto & b) {
        return a.first != b.first ? a.first < b.first : a.second.path < b.second.path;
    });
    for (auto & cur : found) {
        disk.push_back(std::move(cur.second));
    }

    disk_trim(0);

    size_t size = 0;
    for (const auto & e : disk) {
        size += e.size;
    }
    SRV_INF("prompt cache disk tier: '%s', %zu entries, %.3f MiB (limit: %d MiB)\n", dir.c_str(), disk.size(), size / (1024.0 * 1024.0), limit_mib);
}

void server_prompt_cache::evict_front() {
    // shortcut: synchronous write stalls all slots ~0.25 s per GiB, move to a writer thread if evictions get frequent
    if (!disk_dir.empty()) {
        const int64_t t_start = ggml_time_us();
        if (disk_write(states.front())) {
            SRV_INF(" - wrote evicted prompt with %d tokens to disk in %.2f ms\n", states.front().prompt.n_tokens(), (ggml_time_us() - t_start) / 1000.0);
        }
    }

    states.pop_front();
}

void server_prompt_cache::disk_flush() {
    if (disk_dir.empty()) {
        return;
    }

    // skip the oldest entries the size cap would delete again right away
    auto it = states.end();
    size_t size = 0;
    while (it != states.begin()) {
        const size_t size_prev = std::prev(it)->size();
        if (disk_limit > 0 && size + size_prev > disk_limit) {
            break;
        }
        size += size_prev;
        --it;
    }

    // oldest first, so the file mtimes keep the RAM order for the next disk_init()
    int n_written = 0;
    for (; it != states.end(); ++it) {
        n_written += disk_write(*it) ? 1 : 0;
    }

    SRV_INF("prompt cache: wrote %d of %zu RAM entries to disk on shutdown\n", n_written, states.size());
}

void server_prompt_cache::disk_trim(size_t size_new) {
    size_t size = 0;
    for (const auto & e : disk) {
        size += e.size;
    }

    while (!disk.empty() && disk_limit > 0 && size + size_new > disk_limit) {
        SRV_WRN(" - prompt cache disk limit reached, deleting '%s' (%.3f MiB)\n", disk.front().path.c_str(), disk.front().size / (1024.0 * 1024.0));
        std::error_code ec;
        std::filesystem::remove(disk.front().path, ec);
        size -= disk.front().size;
        disk.pop_front();
    }
}

bool server_prompt_cache::disk_write(const server_prompt_cache_state & state) {
    namespace fs = std::filesystem;

    if (!are_lora_equal(state.prompt.lora, disk_lora)) {
        return false; // the file has no lora config
    }

    // an entry restored from disk is evicted again unchanged - the file is still there
    for (const auto & e : disk) {
        if (e.tokens.get_common_prefix(state.prompt.tokens) == state.prompt.tokens.size()) {
            return false;
        }
    }

    std::vector<char> packed;
    try {
        packed = state.prompt.tokens.serialize();
    } catch (const std::exception & e) {
        SRV_WRN(" - cannot write prompt to disk: %s\n", e.what());
        return false;
    }

    // an upper bound of the file size: 1 KiB for the fixed fields, 64 bytes for the fields of each checkpoint
    const size_t size_new = state.size() + packed.size() + disk_fp.size() + 1024 + 64*state.prompt.checkpoints.size();
    if (disk_limit > 0 && size_new > disk_limit) {
        return false;
    }

    disk_trim(size_new);

    uint64_t available = 0;
    if (disk_free) {
        available = disk_free(disk_dir);
    } else {
        std::error_code ec;
        const auto space = fs::space(disk_dir, ec);
        available = ec ? 0 : space.available;
    }
    if (available < size_new + disk_headroom) {
        SRV_WRN(" - not enough free space in '%s' for %.3f MiB, prompt not written\n", disk_dir.c_str(), size_new / (1024.0 * 1024.0));
        return false;
    }

    const uint64_t t_now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const std::string path = (fs::path(disk_dir) / string_format("%016" PRIx64 "-%" PRIu64 ".pcd", t_now, disk_seq++)).string();

    const size_t size_file = disk_write_file(state, packed, path);
    if (size_file == 0) {
        return false;
    }

    disk.push_back({ state.prompt.tokens.clone(), path, size_file });

    return true;
}

size_t server_prompt_cache::disk_write_file(const server_prompt_cache_state & state, const std::vector<char> & packed, const std::string & path) const {
    namespace fs = std::filesystem;

    const std::string path_tmp = path + ".tmp";

    std::error_code ec;
    size_t size_file = 0;
    {
        std::ofstream ofs(path_tmp, std::ios::binary | std::ios::trunc);
        pcd_hash h;
        pcd_write_pod(ofs, h, PCD_MAGIC);
        pcd_write_pod(ofs, h, PCD_VERSION);
        pcd_write_buf(ofs, h, disk_fp);
        h.on = true;
        pcd_write_buf(ofs, h, packed);
        pcd_write_buf(ofs, h, state.data.main);
        pcd_write_buf(ofs, h, state.data.drft);
        pcd_write_pod(ofs, h, (uint32_t) state.prompt.checkpoints.size());
        for (const auto & ckpt : state.prompt.checkpoints) {
            pcd_write_pod(ofs, h, ckpt->n_tokens);
            pcd_write_pod(ofs, h, ckpt->pos_min);
            pcd_write_pod(ofs, h, ckpt->pos_max);
            pcd_write_buf(ofs, h, ckpt->data_tgt);
            pcd_write_buf(ofs, h, ckpt->data_dft);
            pcd_write_buf(ofs, h, ckpt->data_spec);
        }
        h.on = false;
        pcd_write_pod(ofs, h, h.digest());
        ofs.flush();
        size_file = ofs ? (size_t) ofs.tellp() : 0;
        if (!ofs) {
            ofs.close();
            fs::remove(path_tmp, ec);
            SRV_WRN(" - failed to write prompt cache file '%s'\n", path_tmp.c_str());
            return 0;
        }
    }

    fs::rename(path_tmp, path, ec);
    if (ec) {
        fs::remove(path_tmp, ec);
        SRV_WRN(" - failed to rename prompt cache file '%s'\n", path_tmp.c_str());
        return 0;
    }

    return size_file;
}

bool server_prompt_cache::disk_read(const disk_entry & entry, server_prompt_cache_state & state) const {
    std::ifstream ifs(entry.path, std::ios::binary);
    pcd_hash h;
    llama_tokens packed;
    if (!pcd_read_head(ifs, h, disk_fp, packed, entry.size)) {
        return false;
    }

    try {
        state.prompt.tokens = server_tokens::deserialize(packed, disk_mtmd);
        state.prompt.lora   = disk_lora;
    } catch (const std::exception &) {
        return false;
    }

    uint32_t n_ckpt = 0;
    if (!pcd_read_buf(ifs, h, state.data.main, entry.size) ||
        !pcd_read_buf(ifs, h, state.data.drft, entry.size) ||
        !pcd_read_pod(ifs, h, n_ckpt)) {
        return false;
    }

    for (uint32_t i = 0; i < n_ckpt; ++i) {
        auto ckpt = std::make_shared<common_prompt_checkpoint>();
        ckpt->id_task = -1; // not created by a task
        if (!pcd_read_pod(ifs, h, ckpt->n_tokens) ||
            !pcd_read_pod(ifs, h, ckpt->pos_min)  ||
            !pcd_read_pod(ifs, h, ckpt->pos_max)  ||
            !pcd_read_buf(ifs, h, ckpt->data_tgt,  entry.size) ||
            !pcd_read_buf(ifs, h, ckpt->data_dft,  entry.size) ||
            !pcd_read_buf(ifs, h, ckpt->data_spec, entry.size) ||
            ckpt->data_tgt.empty()) {
            return false;
        }
        state.prompt.checkpoints.push_back(std::move(ckpt));
    }

    // the digest closes the file, and the token list must match the file it came from
    const uint64_t digest = h.digest();
    uint64_t digest_file = 0;
    h.on = false;
    return pcd_read_pod(ifs, h, digest_file) && digest_file == digest &&
           ifs.peek() == std::ifstream::traits_type::eof() &&
           state.prompt.tokens.get_common_prefix(entry.tokens) == entry.tokens.size() &&
           state.prompt.tokens.size() == entry.tokens.size();
}

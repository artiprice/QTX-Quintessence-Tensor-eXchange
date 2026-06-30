// ============================================================================
// @file        oom_survival_test.cpp
//@brief OOM Survival Benchmark is the main selling point of the QTX Project.
//
//Demonstrates that due to lithographic quantization
//we can support N times more "agents" in the same
//physical memory than the naive approach.
//
// @author      QTX Project
// @date        2026-05-12
// ============================================================================
//
//SCENARIO:
//1. Create two arenas:
//- HotArena : FP32 KV-porridge agents (full accuracy)
//- ColdArena : INT8 compressed KV-porridge (3.55x compression)
//
//2. We simulate the continuous appearance of agents. Each agent = one
//KV slots in HotArena with a unique data pattern.
//
//3. When HotArena is full, select the oldest active one
//agent (FIFO), quantize it in ColdArena, free up the hot slot.
//
//4. Metrics:
//- active_in_hot : number of agents in FP32
//- active_in_cold : number of agents in INT8
//- total_living_agents : amount (what can work now)
//- hot_bytes_used : physical memory in HotArena
//- cold_bytes_used : physical memory in ColdArena
//- total_bytes_used : total memory
//
//5. Baseline: one FP32 agent = 4 KiB. Without quantization: at 256 KiB
//budget you can keep 64 agents. With quantization: 64 + ~200 =
//~264 agents (ColdArena 256 slots × 1152 bytes = 288 KiB).
//
//HA: benchmark is a separate executable, not a unit test. Uses HA-core
//and quantizer, does not depend on ggml. All stack/heap allocations before start.
// ============================================================================

#include "qtx/arena/fractal_arena.hpp"
#include "qtx/arena/gen_id.hpp"
#include "qtx/quantize/quantizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <span>
#include <vector>

namespace bench {

using namespace qtx;
using core::usize;

// ============================================================================
//Benchmark parameters
// ============================================================================

//Size of one agent KV cache in FP32 bytes.
//4 KiB / sizeof(float) = 1024 elements = 32 blocks × 32 elements.
inline constexpr usize kAgentFP32Bytes = 4096;
inline constexpr usize kAgentElements = kAgentFP32Bytes / sizeof(float);

//HotArena: 64 slots × 4096 = 256 KiB FP32.
using HotArena = arena::FractalArena<64, kAgentFP32Bytes>;
inline constexpr usize kHotBytes = HotArena::kTotalSize;

//ColdArena: 256 slots × 1152 bytes (32 INT8 blocks × 36 = 1152).
//
// EC106 (Phase 2 Block B task 3.1): SlotSize is now sized EXACTLY to the
// compressed payload. Previously we had to round up to 2048 (nearest
// power-of-two ≥ 1152), wasting 896 bytes (43.75%) per slot — for 256
// slots that is 224 KiB of dead padding. Now we use 1152 directly.
// FractalArena no longer requires SlotSize to be a power of two; the
// only remaining constraint is `SlotSize % 128 == 0` (cache-line
// alignment), which 1152 = 9 × 128 satisfies.
inline constexpr usize kColdSlotSize = 1152;
using ColdArena = arena::FractalArena<256, kColdSlotSize>;
inline constexpr usize kColdBytes = ColdArena::kTotalSize;

// ============================================================================
//"Agent" model
// ============================================================================

enum class AgentState : core::u8 {
    kHot,    //lives in HotArena, FP32
    kCold,   //lives in ColdArena, INT8 (quantized)
    kDead,   //completely released
};

struct AgentRecord {
    core::u32   agent_id;           //unique ID, not GenID
    AgentState  state;
    arena::GenID hot_id;            //valid if state == kHot
    arena::GenID cold_id;           //valid if state == kCold
};

//Fill the FP32 slot with a deterministic pattern (for verifiability).
static void fillAgentFP32(std::span<std::byte> view, core::u32 agent_id) noexcept {
    //Pattern: f(i) = sin(agent_id * 0.01 + i * 0.001)
    //— realistic-looking embedding-style values.
    const usize n_elements = view.size() / sizeof(float);
    for (usize i = 0; i < n_elements; ++i) {
        const float v = std::sin(
            static_cast<float>(agent_id) * 0.01f +
            static_cast<float>(i) * 0.001f);
        std::memcpy(view.data() + i * sizeof(float), &v, sizeof(float));
    }
}

//Check that the decompressed data semantically matches the
//original (for INT8: relative error < 5%).
[[nodiscard]] static bool verifyDecompressed(
    std::span<const std::byte> decompressed,
    core::u32 agent_id) noexcept
{
    const usize n_elements = decompressed.size() / sizeof(float);
    for (usize i = 0; i < n_elements; ++i) {
        float v;
        std::memcpy(&v, decompressed.data() + i * sizeof(float), sizeof(float));
        const float expected = std::sin(
            static_cast<float>(agent_id) * 0.01f +
            static_cast<float>(i) * 0.001f);
        if (std::fabs(expected) > 0.05f) {
            const float rel_err = std::fabs(v - expected) / std::fabs(expected);
            if (rel_err > 0.05f) return false;
        }
    }
    return true;
}

// ============================================================================
// Survival Loop
// ============================================================================

struct Stats {
    usize total_agents_created = 0;
    usize total_quantizations = 0;
    usize total_evictions = 0;
    usize peak_active = 0;
    usize peak_hot = 0;
    usize peak_cold = 0;
    usize verify_failures = 0;
    double total_quant_us = 0.0;
};

class SurvivalRun {
public:
    SurvivalRun() noexcept
        : hot_(std::make_unique<HotArena>()),
          cold_(std::make_unique<ColdArena>()) {}

    //Cycle: create an agent → if OOM is hot, quantize the oldest one.
    Stats run(usize target_agents) noexcept {
        Stats s;

        for (usize i = 0; i < target_agents; ++i) {
            createAgent(s);
        }

        //Final verification of all live compressed agents.
        for (auto& rec : agents_) {
            if (rec.state == AgentState::kCold) {
                if (!verifyAgent(rec)) ++s.verify_failures;
            }
        }
        return s;
    }

    [[nodiscard]] usize hotUsed() const noexcept { return hot_->usedSlots(); }
    [[nodiscard]] usize coldUsed() const noexcept { return cold_->usedSlots(); }
    [[nodiscard]] usize hotBytesUsed() const noexcept {
        return hot_->usedSlots() * HotArena::kSlotSize;
    }
    [[nodiscard]] usize coldBytesUsed() const noexcept {
        return cold_->usedSlots() * ColdArena::kSlotSize;
    }

private:
    void createAgent(Stats& s) noexcept {
        const core::u32 new_id = static_cast<core::u32>(agents_.size());

        //1. Try to allocate a hot slot.
        arena::GenID hot_id = hot_->allocate(kAgentFP32Bytes, 128u);

        if (hot_id.isNull()) {
            //HotArena OOM - we are evicting the oldest hot agent.
            if (!evictOldestHot(s)) {
                //ColdArena is also full - there is nothing more we can do.
                return;
            }
            hot_id = hot_->allocate(kAgentFP32Bytes, 128u);
            if (hot_id.isNull()) return;
        }

        //2. Fill in the hot slot.
        auto view = hot_->view(hot_id);
        fillAgentFP32(view, new_id);

        //3. Register an agent.
        agents_.push_back(AgentRecord{
            .agent_id = new_id,
            .state    = AgentState::kHot,
            .hot_id   = hot_id,
            .cold_id  = arena::GenID{},
        });
        hot_queue_.push_back(new_id);

        ++s.total_agents_created;
        s.peak_active = std::max(s.peak_active, activeCount());
        s.peak_hot    = std::max(s.peak_hot, hot_->usedSlots());
        s.peak_cold   = std::max(s.peak_cold, cold_->usedSlots());
    }

    [[nodiscard]] bool evictOldestHot(Stats& s) noexcept {
        //Find the oldest hot agent (FIFO).
        while (!hot_queue_.empty()) {
            const auto candidate_id = hot_queue_.front();
            hot_queue_.pop_front();
            if (candidate_id >= agents_.size()) continue;
            auto& rec = agents_[candidate_id];
            if (rec.state != AgentState::kHot) continue;

            return quantizeToCold(rec, s);
        }
        return false;
    }

    [[nodiscard]] bool quantizeToCold(AgentRecord& rec, Stats& s) noexcept {
        //1. Select a cold slot.
        const auto compressed_bytes =
            quantize::compressedSize(arena::QuantFormat::kINT8, kAgentElements);
        arena::GenID cold_id = cold_->allocate(compressed_bytes, 128u);
        if (cold_id.isNull()) return false;  // ColdArena full

        auto cold_view = cold_->view(cold_id);
        auto hot_view  = hot_->view(rec.hot_id);

        //2. Quantize.
        const auto t0 = std::chrono::steady_clock::now();
        const auto written = quantize::compressFP32ToINT8(
            std::span<const std::byte>(hot_view.data(), kAgentFP32Bytes),
            cold_view);
        const auto t1 = std::chrono::steady_clock::now();
        s.total_quant_us +=
            std::chrono::duration<double, std::micro>(t1 - t0).count();

        if (written != compressed_bytes) {
            //Something went wrong - we roll back the cold allocation.
            (void)cold_->release(cold_id);
            return false;
        }

        //3. Free up the hot slot and update the entry.
        (void)hot_->release(rec.hot_id);
        rec.state   = AgentState::kCold;
        rec.cold_id = cold_id;
        rec.hot_id  = arena::GenID{};
        ++s.total_quantizations;
        ++s.total_evictions;
        return true;
    }

    [[nodiscard]] bool verifyAgent(const AgentRecord& rec) const noexcept {
        if (rec.state != AgentState::kCold) return true;
        auto cold_view = cold_->view(rec.cold_id);
        if (cold_view.empty()) return false;

        //Decompress to local buffer.
        std::vector<std::byte> decompressed(kAgentFP32Bytes);
        const auto sz = quantize::decompressINT8ToFP32(
            std::span<const std::byte>(
                cold_view.data(),
                quantize::compressedSize(arena::QuantFormat::kINT8,
                                         kAgentElements)),
            std::span<std::byte>(decompressed));
        if (sz != kAgentFP32Bytes) return false;
        return verifyDecompressed(decompressed, rec.agent_id);
    }

    [[nodiscard]] usize activeCount() const noexcept {
        return hot_->usedSlots() + cold_->usedSlots();
    }

    std::unique_ptr<HotArena>  hot_;
    std::unique_ptr<ColdArena> cold_;
    std::vector<AgentRecord>   agents_;
    std::deque<core::u32>      hot_queue_;  // FIFO: agent IDs in hot order
};

// ============================================================================
//Report
// ============================================================================

static void printHeader() {
    std::printf("\n");
    std::printf("╔════════════════════════════════════════════════════════════════╗\n");
    std::printf("║       QTX — OOM Survival Benchmark                          ║\n");
    std::printf("╠════════════════════════════════════════════════════════════════╣\n");
    std::printf("║ HotArena:  %3zu slots × %5zu B = %6zu KiB FP32                ║\n",
                HotArena::kSlotCount, HotArena::kSlotSize,
                kHotBytes / 1024u);
    std::printf("║ ColdArena: %3zu slots × %5zu B = %6zu KiB INT8                ║\n",
                ColdArena::kSlotCount, ColdArena::kSlotSize,
                kColdBytes / 1024u);
    std::printf("║ Total physical memory: %6zu KiB                              ║\n",
                (kHotBytes + kColdBytes) / 1024u);
    std::printf("╚════════════════════════════════════════════════════════════════╝\n");
}

static void printResult(usize target, const SurvivalRun& run, const Stats& s) {
    const usize active = run.hotUsed() + run.coldUsed();
    const usize bytes_used = run.hotBytesUsed() + run.coldBytesUsed();
    const float bytes_per_agent_naive =
        static_cast<float>(kAgentFP32Bytes);
    const float bytes_per_agent_actual =
        active > 0 ? static_cast<float>(bytes_used) / static_cast<float>(active)
                   : 0.0f;
    const float density_gain =
        active > 0 ? bytes_per_agent_naive / bytes_per_agent_actual : 0.0f;

    std::printf("\n");
    std::printf("──── Target: %zu agents ────\n", target);
    std::printf("  Created:        %zu\n", s.total_agents_created);
    std::printf("  Active (total): %zu  (hot: %zu, cold: %zu)\n",
                active, run.hotUsed(), run.coldUsed());
    std::printf("  Peak active:    %zu\n", s.peak_active);
    std::printf("  Quantizations:  %zu  (avg %.2f μs/op)\n",
                s.total_quantizations,
                s.total_quantizations > 0
                    ? s.total_quant_us / static_cast<double>(s.total_quantizations)
                    : 0.0);
    std::printf("  Bytes used:     %zu KiB / %zu KiB  (%.1f%%)\n",
                bytes_used / 1024u, (kHotBytes + kColdBytes) / 1024u,
                static_cast<double>(bytes_used) /
                    static_cast<double>(kHotBytes + kColdBytes) * 100.0);
    std::printf("  Bytes/agent:    %.0f naive → %.0f actual  (%.2fx density)\n",
                static_cast<double>(bytes_per_agent_naive),
                static_cast<double>(bytes_per_agent_actual),
                static_cast<double>(density_gain));
    std::printf("  Verify fails:   %zu\n", s.verify_failures);
}

}  // namespace bench

// ============================================================================
// Entry point
// ============================================================================

int main() {
    bench::printHeader();

    //Several targets to see how the system scales.
    for (auto target : {64u, 128u, 256u, 320u}) {
        bench::SurvivalRun run;
        const auto t0 = std::chrono::steady_clock::now();
        const auto stats = run.run(target);
        const auto t1 = std::chrono::steady_clock::now();

        bench::printResult(target, run, stats);
        std::printf("  Wall time:      %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count());

        if (stats.verify_failures > 0) {
            std::printf("\n*** VERIFY FAILURES — quantization is lossy! ***\n");
            return 1;
        }
    }

    std::printf("\n");
    std::printf("╔════════════════════════════════════════════════════════════════╗\n");
    std::printf("║  BENCHMARK COMPLETE — all agents successfully verified.        ║\n");
    std::printf("╚════════════════════════════════════════════════════════════════╝\n");
    return 0;
}

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/sgpr_layout.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/runtime_wait_target.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {
class Instruction;
namespace amdgpu {
class Wavefront;
enum class WaitCounterType : uint8_t;
struct RegisterModifiers;

/// A per-wave register filter owned exclusively by the instruction issuer.
/// Instruction-boundary validation completes before async work is published;
/// helper threads and plugin observers never access this filter.
class MemoryWaitShadow {
public:
  // Reserve scalar encoding slots separately from allocatable SGPRs. ACC
  // registers are not destinations of the modeled memory producers.
  static constexpr size_t kSgprEncodingSlots = 128;
  static constexpr size_t kTtmpBase = REGISTER_SET_MAX_VGPRS + kSgprEncodingSlots;
  static constexpr size_t kSpecialBase = kTtmpBase + REGISTER_SET_MAX_TTMPS;
  static constexpr size_t kScalarShadowSlots = 256;
  static constexpr size_t kRegisters = REGISTER_SET_MAX_VGPRS + kScalarShadowSlots;
  /// Map the named gfx942 special registers into the SGPR dependence space.
  void set_sgpr_alias_end(uint16_t end) { sgpr_alias_end_ = end; }
  RegisterRef canonical(RegisterRef reg) const {
    if (sgpr_alias_end_) {
      if (reg.cls == RegClass::VCC)
        return {RegClass::SGPR, static_cast<uint16_t>(cdna3::vcc_base(sgpr_alias_end_) + reg.index),
                reg.width};
      if (reg.cls == RegClass::FLAT_SCRATCH)
        return {RegClass::SGPR,
                static_cast<uint16_t>(cdna3::flat_scratch_base(sgpr_alias_end_) + reg.index),
                reg.width};
    }
    return reg;
  }
  static size_t index(RegisterRef reg) {
    if (reg.cls == RegClass::VGPR && reg.index < REGISTER_SET_MAX_VGPRS)
      return reg.index;
    if (reg.cls == RegClass::SGPR && reg.index < kSgprEncodingSlots)
      return REGISTER_SET_MAX_VGPRS + reg.index;
    if (reg.cls == RegClass::TTMP && reg.index < REGISTER_SET_MAX_TTMPS)
      return kTtmpBase + reg.index;
    if (is_special_reg_class(reg.cls))
      return kSpecialBase + static_cast<size_t>(reg.cls);
    return kRegisters;
  }
  static constexpr uint8_t kResult = 1;
  static constexpr uint8_t kSource = 2;
  bool test(size_t i, bool write = false) const {
    return i < kRegisters && (bytes_[i] & (write ? kResult | kSource : kResult));
  }
  bool pending_vgpr(uint16_t index, uint8_t width) const {
    assert(index + width <= REGISTER_SET_MAX_VGPRS);
    if (width == 1)
      return bytes_[index] != 0;
    for (unsigned i = 0; i < width; ++i)
      if (bytes_[index + i])
        return true;
    return false;
  }

  bool pending(RegisterRef reg, bool write = false) const {
    reg = canonical(reg);
    for (unsigned r = 0; r < reg.width; ++r) {
      auto element = reg;
      element.index += r;
      if (test(index(element), write))
        return true;
    }
    return false;
  }
  void set(size_t i, uint8_t bits = kResult) { bytes_[i] |= bits; }
  void clear(size_t i) { bytes_[i] = 0; }

  // Result ranges are normally contiguous physical VGPRs. Resolve their file
  // once and combine the overlap probe with marking the same shadow bytes.
  bool mark(RegisterRef reg, uint8_t bits) {
    reg = canonical(reg);
    bool overlap = false;
    if (reg.cls == RegClass::VGPR && reg.index + reg.width <= REGISTER_SET_MAX_VGPRS) {
      for (unsigned r = 0; r < reg.width; ++r) {
        auto &byte = bytes_[reg.index + r];
        overlap |= byte != 0;
        byte |= bits;
      }
    } else {
      for (unsigned r = 0; r < reg.width; ++r) {
        auto element = reg;
        element.index += r;
        if (const auto i = index(element); i < kRegisters) {
          overlap |= bytes_[i] != 0;
          bytes_[i] |= bits;
        }
      }
    }
    return overlap;
  }
  void clear(RegisterRef reg) {
    reg = canonical(reg);
    if (reg.cls == RegClass::VGPR && reg.index + reg.width <= REGISTER_SET_MAX_VGPRS) {
      for (unsigned r = 0; r < reg.width; ++r)
        bytes_[reg.index + r] = 0;
    } else {
      for (unsigned r = 0; r < reg.width; ++r) {
        auto element = reg;
        element.index += r;
        if (const auto i = index(element); i < kRegisters)
          bytes_[i] = 0;
      }
    }
  }
  void reset() { bytes_.fill(0); }

private:
  uint16_t sgpr_alias_end_ = 0;
  std::array<uint8_t, kRegisters> bytes_{};
};

/// Tracks software-visible memory dependencies independently of eager writeback.
/// No memory payloads, decoder objects, or helper threads are retained here.
class MemoryWaitScoreboard {
public:
  static constexpr uint8_t kFullDwordByteMask = 0xf;
  static constexpr uint16_t kUnordered = UINT16_MAX;
  explicit MemoryWaitScoreboard(MemoryWaitShadow &shadow) : pending_(shadow) {
    last_order_.fill(kUnordered);
    last_capacity_eligible_.fill(false);
  }
  struct Event {
    uint64_t sequence;
    uint64_t pc;
    uint64_t lanes;
    RegisterRef reg;
    WaitCounterKind counter;
    uint8_t bytes;
    bool reported = false;
    uint16_t order = kUnordered;
    uint64_t order_sequence = 0;
    uint64_t capacity_order_sequence = 0;
    // Whether this pending result can be overwritten without a wait by a later
    // load in the same order class. That also requires the incoming load to
    // qualify for ordered writeback (see access_pending). Results whose counter
    // decrements in order are not necessarily written to registers in order.
    bool ordered_overwrite = true;
    bool export_source = false;
    bool valu_source = false;
    uint8_t wmma_wait_states = 0;
    uint64_t preceding_core_valu = 0;
    // An async transfer's possible replay source uses its ASYNC completion
    // position. It contributes no proven credit to the ordered X queue.
    bool async_replay_source = false;
    bool atomic_replay_source = false;
    // Only an active, actually issued native GFX10 DS request can release its
    // scalar source lock when the DS completion counter fully drains.
    bool gfx10_native_ds_source = false;
    bool gfx10_native_ds_issued = false;
    bool protects_replay_source() const {
      return counter == WaitCounterKind::X || async_replay_source;
    }
    bool protects_source() const {
      return protects_replay_source() || export_source || counter == WaitCounterKind::VmVsrc ||
             valu_source;
    }
    uint8_t shadow_bits() const {
      return protects_source() ? MemoryWaitShadow::kSource : MemoryWaitShadow::kResult;
    }
  };
  struct Hazard {
    Event producer;
    uint64_t consumer_pc;
    RegisterRef reg;
    bool write;
    uint32_t required_wait;
  };
  using Reporter = void (*)(void *, const Hazard &);

  bool empty() const { return events_.empty(); }
  void clear();
  /// Cancel the latest instruction's dependency and counter positions before
  /// debugger replay, preserving older producers and coalesced source records.
  bool cancel_last_issue(uint64_t pc, std::span<const WaitCounterType> counters,
                         rj_code_arch_t arch, bool xcnt, bool vm_vsrc = false);
  struct IssueSnapshot {
    std::array<uint64_t, static_cast<size_t>(WaitCounterKind::Count)> issued;
  };
  /// Capture positions before tracking one instruction. No further instruction
  /// may issue before cancelling that instruction with this snapshot.
  IssueSnapshot issue_snapshot() const { return {issued_}; }
  bool cancel_last_issue(uint64_t pc, const IssueSnapshot &before);
  bool issues_flat_counter(const Instruction &inst, Wavefront &wf, uint64_t requests,
                           uint64_t returned_without_request, WaitCounterKind counter) const;
  /// Resolve register-relative waits from the actual wave state.
  [[nodiscard]] static util::FailureOr<std::optional<WaitFields>>
  explicit_wait_fields(const Instruction &inst, rj_code_arch_t arch, Wavefront *wf);
  void before(const Instruction &inst, rj_code_arch_t arch, Wavefront *wf = nullptr);
  void track_gfx10_scalar_sources(const Instruction &inst, Wavefront &wf,
                                  const waitcheck_detail::ClassifiedEvent &source);
  void confirm_gfx10_native_ds_source_issue(uint64_t pc);
  /// Return whether address preparation can change which lanes supply VDATA.
  static bool needs_buffer_payload_lanes(const Instruction &inst, rj_code_arch_t arch);
  void check_instruction(const Instruction &inst, Wavefront &wf);
  struct SourceRoute {
    WaitCounterKind counter = WaitCounterKind::Count;
    uint64_t sequence = 0;
    uint64_t lanes = 0;
    uint16_t order = kUnordered;
    uint64_t order_sequence = 0;
    uint64_t capacity_order_sequence = 0;
  };
  [[nodiscard]] SourceRoute source_route(WaitCounterKind counter, uint64_t sequence,
                                         uint64_t lanes) const;
  void track_expert_vm_sources(const Instruction &inst, Wavefront &wf,
                               const waitcheck_detail::ClassifiedEvent &event,
                               std::span<const SourceRoute> routes);
  void leave_expert_scheduling();
  void track_expert_valu(const Instruction &inst, Wavefront &wf,
                         waitcheck_detail::WaitEventKind kind, rj_code_target_id_t target);
  struct MemoryLanes {
    uint64_t issue;
    uint64_t result;
  };
  static MemoryLanes memory_lanes(const Instruction &inst, Wavefront &wf);
  static uint64_t result_lanes(const Instruction &inst, Wavefront &wf);
  static bool result_is_written(const Instruction &inst, Wavefront &wf);
  struct FlatLanes {
    uint64_t requests;
    uint64_t shared;
    uint64_t private_oob_results;
  };
  static FlatLanes flat_lanes(const Instruction &inst, Wavefront &wf, uint64_t shared_base,
                              uint64_t shared_limit);
  /// X has one translation group at a time, independent of completion queues.
  void check_prefetch_xcnt() const;
  void check_atomic_xcnt() const;
  void xcnt_group(bool scalar);
  /// Map translation to this instruction's completion position, if it received one.
  uint64_t issue_xcnt(std::optional<WaitCounterKind> completion, bool scalar);
  /// A VMEM destination orders translation of older overlapping VMEM sources.
  void xcnt_ordered_write(RegisterRef reg, uint64_t lanes, uint8_t bytes);
  /// Count every operation, including stores without a register destination.
  uint64_t issue(WaitCounterKind counter, bool unordered = false, uint32_t ordering_kind = 0,
                 uint32_t units = 1, bool capacity_eligible = true);
  uint64_t issue(const waitcheck_detail::ClassifiedEvent &event, rj_code_arch_t arch,
                 uint32_t units = 1, bool capacity_eligible = true);
  /// Use the same decoded counter units for admission and issue accounting.
  static uint32_t issue_units(const Instruction &inst,
                              const waitcheck_detail::ClassifiedEvent &event, rj_code_arch_t arch);
  /// Find the incoming producer's FIFO class without issuing its completion.
  uint16_t ordered_write_order(const waitcheck_detail::ClassifiedEvent &event,
                               rj_code_arch_t arch) const;
  /// Admission can force completion before the incoming instruction reads its operands.
  void backpressure(WaitCounterKind counter, uint32_t capacity, uint32_t incoming_units = 1);
  void add(Event event);
  void wait(WaitCounterKind counter, uint32_t threshold);
  uint64_t outstanding(WaitCounterKind counter) const {
    const auto i = static_cast<size_t>(counter);
    return issued_[i] - retired_[i];
  }
  [[gnu::always_inline]] void access(RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write,
                                     uint16_t ordered_write_order = kUnordered) {
    if (!lanes || !bytes)
      return;
    if (!pending_.pending(reg, write))
      return;
    access_pending(pending_.canonical(reg), lanes, bytes, write, ordered_write_order);
  }

  void bind(uint64_t pc, void *context, Reporter reporter) {
    pc_ = pc;
    context_ = context;
    reporter_ = reporter;
  }
  const std::vector<Event> &events() const { return events_; }

private:
  static constexpr size_t kCounters = static_cast<size_t>(WaitCounterKind::Count);
  bool cancel_issue_units(uint64_t pc, const std::array<uint32_t, kCounters> &units);
  static constexpr size_t kRegisters = MemoryWaitShadow::kRegisters;
  static size_t index(RegisterRef reg) { return MemoryWaitShadow::index(reg); }

  template <typename Visit>
  void visit_register_accesses(const Instruction &inst, Wavefront &wf, Visit &&visit,
                               bool check_result = true);
  [[gnu::cold]] RJ_NOINLINE void check_instruction_pending(const Instruction &inst, Wavefront &wf);
  void check_memory_result(const Instruction &inst, Wavefront &wf, uint64_t vector_lanes,
                           const RegisterModifiers &modifiers);
  void access_pending(RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write,
                      uint16_t ordered_write_order);
  void rebuild_mask();
  void clear_destination(const Event &event);
  void retire_completed();
  void wait_ordered_xcnt(uint32_t threshold);
  bool drain_async_replay();
  bool apply_wait(WaitCounterKind counter, uint32_t threshold);
  bool completed(WaitCounterKind counter, uint64_t sequence, uint16_t order,
                 uint64_t order_sequence, uint64_t capacity_order_sequence) const;
  void stamp_order(Event &event) const;
  uint32_t required_wait(const Event &event) const;
  std::array<uint64_t, kCounters> issued_{};
  std::array<uint64_t, kCounters> retired_{};
  std::array<bool, kCounters> unordered_{};
  std::array<uint32_t, kCounters> ordering_kinds_{};
  struct Order {
    WaitCounterKind counter;
    uint32_t kind;
    uint64_t issued = 0;
    uint64_t retired = 0;
    uint64_t capacity_issued = 0;
    uint64_t capacity_retired = 0;
  };
  // Counter membership is not completion order. Only operations in the same
  // ordered class can prove that an old result must have completed.
  const Order *expert_core_valu_order() const;
  std::vector<Order> orders_;
  std::array<uint16_t, kCounters> last_order_{};
  std::array<bool, kCounters> last_capacity_eligible_{};
  bool xcnt_scalar_ = false;
  uint64_t async_replay_drained_ = 0;
  struct Translation {
    uint64_t sequence;
    uint64_t completion_sequence;
    WaitCounterKind completion;
    uint16_t order;
    uint64_t order_sequence;
    uint64_t capacity_order_sequence;
  };
  std::vector<Translation> translations_;
  std::map<uint64_t, std::array<SourceRoute, 2>> expert_source_completions_;
  bool checking_valu_ = false;
  bool pending_scalar_ = false;
  // Once two events share a shadow slot, retirement must restore overlapping
  // entries until the wave has no pending events. Lane/byte disjointness does
  // not make their shared register byte independently clearable.
  bool may_overlap_ = false;
  MemoryWaitShadow &pending_;
  std::vector<Event> events_;
  struct SupersededEvent {
    Event event;
    WaitCounterKind replacement_counter;
    uint64_t replacement_sequence;
    uint64_t replacement_pc;
  };
  // Keep predecessors of only the newest coalescing issue in each counter,
  // without adding history work to the usual non-overlapping issue path.
  std::vector<SupersededEvent> superseded_;
  uint64_t pc_ = 0;
  void *context_ = nullptr;
  Reporter reporter_ = nullptr;
};

} // namespace amdgpu
} // namespace rocjitsu

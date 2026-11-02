// Bounded, process-local diagnostic. No system capture or persistent changes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

struct Record {
  uint64_t tag, sequence, commit_ns;
  double gpu_start, gpu_end;
  uint64_t status;
};
static std::atomic<uint64_t> active{0}, pending{0}, sequence{0}, dropped{0};
static std::mutex guard;
static std::vector<Record> records;
static IMP original = nullptr;
static Class observed_class = Nil;
static constexpr size_t max_records = 8192;

static void observed_commit(id object, SEL selector) {
  const uint64_t tag = active.load();
  if (tag) {
    const uint64_t number = sequence.fetch_add(1);
    const uint64_t before = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    pending.fetch_add(1);
    [(id<MTLCommandBuffer>)object addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
      Record row{tag, number, before, buffer.GPUStartTime, buffer.GPUEndTime,
                 static_cast<uint64_t>(buffer.status)};
      {
        std::lock_guard<std::mutex> lock(guard);
        if (records.size() < max_records) records.push_back(row);
        else dropped.fetch_add(1);
      }
      pending.fetch_sub(1);
    }];
  }
  reinterpret_cast<void (*)(id, SEL)>(original)(object, selector);
}

extern "C" int kd_install() {
  @autoreleasepool {
    if (original) return -1;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return -2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBufferWithUnretainedReferences];
    observed_class = object_getClass(buffer);
    Method method = class_getInstanceMethod(observed_class, @selector(commit));
    if (!method) return -3;
    original = method_getImplementation(method);
    class_replaceMethod(observed_class, @selector(commit),
                        reinterpret_cast<IMP>(observed_commit), method_getTypeEncoding(method));
    records.reserve(max_records);
    return 0;
  }
}

extern "C" const char* kd_class() { return class_getName(observed_class); }
extern "C" int kd_uninstall() {
  if (active.load() || pending.load() || !original) return -1;
  class_replaceMethod(observed_class, @selector(commit), original, "v@:");
  original = nullptr;
  return 0;
}
extern "C" uint64_t kd_pending() { return pending.load(); }
extern "C" uint64_t kd_dropped() { return dropped.load(); }
extern "C" uint64_t kd_committed() { return sequence.load(); }
extern "C" int kd_begin(uint64_t tag) {
  if (!tag || active.load() || pending.load()) return -1;
  { std::lock_guard<std::mutex> lock(guard); records.clear(); }
  active.store(tag);
  return 0;
}
extern "C" void kd_end() { active.store(0); }
extern "C" size_t kd_read(Record* destination, size_t capacity) {
  std::lock_guard<std::mutex> lock(guard);
  if (capacity < records.size()) return max_records + 1;
  std::copy(records.begin(), records.end(), destination);
  return records.size();
}

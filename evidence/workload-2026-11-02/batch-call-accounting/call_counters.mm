// Fixed own-process Metal API counters. No system capture or persistent hook.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

struct Event { uint64_t tag, kind, a, b, c, d, e, f; char name[160]; };
struct Hook { Class cls; SEL sel; IMP original; std::string types; };
static std::mutex guard;
static std::vector<Event> events;
static std::vector<Hook> hooks;
static std::unordered_map<void*,std::string> pipelines, encoders;
static std::atomic<uint64_t> active{0},pending{0},dropped{0};
static constexpr size_t capacity=4096;
static Class device_class=Nil;
static thread_local unsigned allocation_depth=0,dispatch_depth=0;

static IMP original(id object,SEL selector) {
  std::lock_guard<std::mutex> lock(guard);
  for(Class cls=object_getClass(object);cls;cls=class_getSuperclass(cls))
    for(const auto& hook:hooks)if(hook.cls==cls && hook.sel==selector)return hook.original;
  std::abort();
}
static void emit(uint64_t tag,uint64_t kind,uint64_t a=0,uint64_t b=0,uint64_t c=0,
                 uint64_t d=0,uint64_t e=0,uint64_t f=0,const std::string& name="") {
  if(!tag)return;
  Event event{tag,kind,a,b,c,d,e,f,{}};
  if(name.size()>=sizeof(event.name)){dropped.fetch_add(1);return;}
  std::memcpy(event.name,name.data(),name.size());
  std::lock_guard<std::mutex> lock(guard);
  if(events.size()<capacity)events.push_back(event);else dropped.fetch_add(1);
}
static void remember(id pipeline,id<MTLFunction> function) {
  if(!pipeline)return;
  const char* text=function.name.UTF8String;
  std::lock_guard<std::mutex> lock(guard);
  if(pipelines.size()>=512){dropped.fetch_add(1);return;}
  pipelines[(void*)pipeline]=text?text:"unknown";
}
static id pipeline_function(id self,SEL sel,id function,void* error) {
  id result=((id(*)(id,SEL,id,void*))original(self,sel))(self,sel,function,error);
  remember(result,function);return result;
}
static id pipeline_descriptor(id self,SEL sel,MTLComputePipelineDescriptor* descriptor,
                              NSUInteger options,void* reflection,void* error) {
  id result=((id(*)(id,SEL,id,NSUInteger,void*,void*))original(self,sel))(
      self,sel,descriptor,options,reflection,error);
  remember(result,descriptor.computeFunction);return result;
}
static id buffer_length(id self,SEL sel,NSUInteger length,MTLResourceOptions options) {
  const bool outer=allocation_depth++==0;
  id<MTLBuffer> result=((id(*)(id,SEL,NSUInteger,MTLResourceOptions))original(self,sel))(self,sel,length,options);
  --allocation_depth;if(outer)emit(active.load(),1,length,result?result.storageMode:99,
       object_getClass(self)==device_class?1:2,result?result.length:0);return result;
}
static id buffer_bytes(id self,SEL sel,const void* bytes,NSUInteger length,MTLResourceOptions options) {
  const bool outer=allocation_depth++==0;
  id<MTLBuffer> result=((id(*)(id,SEL,const void*,NSUInteger,MTLResourceOptions))original(self,sel))(self,sel,bytes,length,options);
  --allocation_depth;if(outer)emit(active.load(),2,length,result?result.storageMode:99,1,result?result.length:0);return result;
}
static id buffer_no_copy(id self,SEL sel,void* bytes,NSUInteger length,MTLResourceOptions options,id deallocator) {
  const bool outer=allocation_depth++==0;
  id<MTLBuffer> result=((id(*)(id,SEL,void*,NSUInteger,MTLResourceOptions,id))original(self,sel))(self,sel,bytes,length,options,deallocator);
  --allocation_depth;if(outer)emit(active.load(),3,length,result?result.storageMode:99,1,result?result.length:0);return result;
}
static void copy_buffer(id self,SEL sel,id source,NSUInteger source_offset,id destination,
                        NSUInteger destination_offset,NSUInteger size) {
  emit(active.load(),4,size,source_offset,destination_offset);
  ((void(*)(id,SEL,id,NSUInteger,id,NSUInteger,NSUInteger))original(self,sel))(
      self,sel,source,source_offset,destination,destination_offset,size);
}
static void set_pipeline(id self,SEL sel,id pipeline) {
  if(active.load()) {
    std::lock_guard<std::mutex> lock(guard);
    if(encoders.size()>=128)dropped.fetch_add(1);
    else encoders[(void*)self]=pipelines.count((void*)pipeline)?pipelines[(void*)pipeline]:"unknown";
  }
  ((void(*)(id,SEL,id))original(self,sel))(self,sel,pipeline);
}
static std::string kernel_name(id encoder) {
  std::lock_guard<std::mutex> lock(guard);
  auto item=encoders.find((void*)encoder);return item==encoders.end()?"unknown":item->second;
}
static void dispatch_groups(id self,SEL sel,MTLSize grid,MTLSize group) {
  const bool outer=dispatch_depth++==0;
  if(outer)emit(active.load(),5,grid.width,grid.height,grid.depth,group.width,group.height,group.depth,kernel_name(self));
  ((void(*)(id,SEL,MTLSize,MTLSize))original(self,sel))(self,sel,grid,group);
  --dispatch_depth;
}
static void dispatch_threads(id self,SEL sel,MTLSize grid,MTLSize group) {
  const bool outer=dispatch_depth++==0;
  if(outer)emit(active.load(),6,grid.width,grid.height,grid.depth,group.width,group.height,group.depth,kernel_name(self));
  ((void(*)(id,SEL,MTLSize,MTLSize))original(self,sel))(self,sel,grid,group);
  --dispatch_depth;
}
static void commit(id self,SEL sel) {
  const uint64_t tag=active.load();
  if(tag) {
    emit(tag,7);pending.fetch_add(1);
    [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> buffer){
      emit(tag,8,buffer.status);pending.fetch_sub(1);
    }];
  }
  ((void(*)(id,SEL))original(self,sel))(self,sel);
}
static bool install(id object,SEL sel,IMP replacement) {
  Class cls=object_getClass(object);
  for(const auto& hook:hooks)if(hook.cls==cls && hook.sel==sel)return true;
  Method method=class_getInstanceMethod(cls,sel);if(!method)return false;
  hooks.push_back({cls,sel,method_getImplementation(method),method_getTypeEncoding(method)});
  class_replaceMethod(cls,sel,replacement,method_getTypeEncoding(method));return true;
}
static void restore() {
  for(auto it=hooks.rbegin();it!=hooks.rend();++it)class_replaceMethod(it->cls,it->sel,it->original,it->types.c_str());
  hooks.clear();
}
extern "C" int kd_install() {
  @autoreleasepool {
    if(!hooks.empty())return -1;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device)return -2;
    device_class=object_getClass(device);
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLCommandBuffer> cb=[queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> encoder=[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [encoder endEncoding];id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];[blit endEncoding];
    MTLHeapDescriptor* desc=[[[MTLHeapDescriptor alloc]init]autorelease];
    desc.size=1024*1024;desc.storageMode=MTLStorageModeShared;
    id<MTLHeap> heap=[device newHeapWithDescriptor:desc];
    bool ok=heap && install(device,@selector(newBufferWithLength:options:),(IMP)buffer_length)
      && install(heap,@selector(newBufferWithLength:options:),(IMP)buffer_length)
      && install(device,@selector(newBufferWithBytes:length:options:),(IMP)buffer_bytes)
      && install(device,@selector(newBufferWithBytesNoCopy:length:options:deallocator:),(IMP)buffer_no_copy)
      && install(device,@selector(newComputePipelineStateWithFunction:error:),(IMP)pipeline_function)
      && install(device,@selector(newComputePipelineStateWithDescriptor:options:reflection:error:),(IMP)pipeline_descriptor)
      && install(encoder,@selector(setComputePipelineState:),(IMP)set_pipeline)
      && install(encoder,@selector(dispatchThreadgroups:threadsPerThreadgroup:),(IMP)dispatch_groups)
      && install(encoder,@selector(dispatchThreads:threadsPerThreadgroup:),(IMP)dispatch_threads)
      && install(blit,@selector(copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:),(IMP)copy_buffer)
      && install(cb,@selector(commit),(IMP)commit);
    events.reserve(capacity);pipelines.reserve(512);encoders.reserve(128);
    [heap release];[queue release];[device release];
    if(!ok){restore();return -3;}return 0;
  }
}
extern "C" int kd_begin(uint64_t tag) {
  if(!tag || active.load() || pending.load() || hooks.empty())return -1;
  {std::lock_guard<std::mutex> lock(guard);events.clear();encoders.clear();}
  active.store(tag);return 0;
}
extern "C" void kd_end(){active.store(0);}
extern "C" uint64_t kd_pending(){return pending.load();}
extern "C" uint64_t kd_dropped(){return dropped.load();}
extern "C" size_t kd_read(Event* destination,size_t n) {
  std::lock_guard<std::mutex> lock(guard);if(n<events.size())return capacity+1;
  std::copy(events.begin(),events.end(),destination);return events.size();
}
extern "C" int kd_uninstall() {
  if(active.load() || pending.load() || hooks.empty())return -1;restore();return 0;
}
// Positive control for exact allocation/copy counts before application use.
extern "C" int kd_copy_control() {
  @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> source=[device newBufferWithLength:128 options:MTLResourceStorageModeShared];
    id<MTLBuffer> destination=[device newBufferWithLength:128 options:MTLResourceStorageModeShared];
    char bytes[32]={};id<MTLBuffer> copied=[device newBufferWithBytes:bytes length:32 options:MTLResourceStorageModeShared];
    const size_t page=sysconf(_SC_PAGESIZE);void* memory=nullptr;
    if(posix_memalign(&memory,page,page))return -2;
    id<MTLBuffer> wrapped=[device newBufferWithBytesNoCopy:memory length:page options:MTLResourceStorageModeShared deallocator:nil];
    MTLHeapDescriptor* desc=[[[MTLHeapDescriptor alloc]init]autorelease];desc.size=1024*1024;desc.storageMode=MTLStorageModeShared;
    id<MTLHeap> heap=[device newHeapWithDescriptor:desc];id<MTLBuffer> small=[heap newBufferWithLength:64 options:MTLResourceStorageModeShared];
    NSError* error=nil;
    NSString* code=@"#include <metal_stdlib>\nusing namespace metal; kernel void kd_counter_control(device uint* x [[buffer(0)]],uint i [[thread_position_in_grid]]) {if(i==0)x[0]=19u;}";
    id<MTLLibrary> library=[device newLibraryWithSource:code options:nil error:&error];
    id<MTLFunction> function=[library newFunctionWithName:@"kd_counter_control"];
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    if(!source || !destination || !copied || !wrapped || !small || !pipeline)return -3;
    std::memset(source.contents,37,128);std::memset(destination.contents,0,128);
    id<MTLCommandBuffer> cb=[queue commandBufferWithUnretainedReferences];id<MTLBlitCommandEncoder> encoder=[cb blitCommandEncoder];
    [encoder copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:16];[encoder endEncoding];
    id<MTLComputeCommandEncoder> compute=[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [compute setComputePipelineState:pipeline];[compute setBuffer:small offset:0 atIndex:0];
    [compute dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [compute memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [compute dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[compute endEncoding];
    [cb commit];[cb waitUntilCompleted];bool ok=cb.status==MTLCommandBufferStatusCompleted;
    for(int i=0;i<128;i++)ok&=((unsigned char*)destination.contents)[i]==(i<16?37:0);
    ok&=((uint32_t*)small.contents)[0]==19u;
    [pipeline release];[function release];[library release];[small release];[heap release];
    [copied release];[wrapped release];free(memory);
    [source release];[destination release];[queue release];[device release];return ok?0:-1;
  }
}

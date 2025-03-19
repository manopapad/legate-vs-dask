#include <legate.h>

#include <nccl.h>

namespace lg = legate;
using legate::Memory::Kind;

struct NCCLTester : public legate::LegateTask<NCCLTester> {
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{0}};
  static constexpr auto GPU_VARIANT_OPTIONS =
    lg::VariantOptions{}
    .with_concurrent(true)
    .with_has_allocations(true);
    .with_communicators({"nccl"});

  static void gpu_variant(legate::TaskContext context)
  {
    auto comm = context.communicators().at(0).get<ncclComm_t*>();
    size_t N = context.get_launch_domain().get_volume();

    auto recv_buf =
      lg::create_buffer<uint64_t>(N, Kind::Z_COPY_MEM).ptr(0);
    auto send_buf =
      lg::create_buffer<uint64_t>(1, Kind::Z_COPY_MEM).ptr(0);

    auto* p_recv = recv_buf.ptr(0);
    auto* p_send = send_buf.ptr(0);

    *p_send = 12345;

    auto stream = context.get_task_stream();
    ncclAllGather(p_send, p_recv, 1, ncclUint64, *comm, stream);
    cudaStreamSynchronize(stream);

    for (std::uint32_t idx = 0; idx < N; ++idx)
      assert(p_recv[idx] == 12345);
  }
};

int main() {
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  if (machine.count(legate::mapping::TaskTarget::GPU) < 2) {
    return;
  }

  auto library = runtime->create_library("test_nccl", legate::ResourceConfig{},
                                         nullptr, {});
  NCCLTester::register_variants(library);

  auto store = runtime->create_store(legate::full(1, SIZE), legate::int32());
  auto task = runtime->create_task(context, NCCLTester::TASK_CONFIG.task_id());
  task.add_output(store);
  runtime->submit(std::move(task));
}

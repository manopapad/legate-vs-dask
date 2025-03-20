#include <iostream>
#include <legate.h>
#include <nccl.h>

namespace lg = legate;

constexpr auto ZCMEM = lg::Memory::Kind::Z_COPY_MEM;

struct NCCLTester : public lg::LegateTask<NCCLTester> {

  static inline const auto TASK_CONFIG = lg::TaskConfig{lg::LocalTaskID{0}};
  static inline const auto GPU_VARIANT_OPTIONS =
    lg::VariantOptions{}.with_concurrent(true).with_has_allocations(true).with_communicators({"nccl"});

  static void gpu_variant(lg::TaskContext context)
  {
    auto comm = context.communicators().at(0).get<ncclComm_t*>();
    size_t N = context.get_launch_domain().get_volume();

    auto recv_buf = lg::create_buffer<uint64_t>(N, ZCMEM);
    auto send_buf = lg::create_buffer<uint64_t>(1, ZCMEM);

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

int main()
{
  lg::start();

  auto runtime = lg::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  if (machine.count(lg::mapping::TaskTarget::GPU) < 2) {
    std::cerr << "Need at least 2 GPUs" << std::endl;
    return 1;
  }

  auto library = runtime->create_library("test_nccl", lg::ResourceConfig{}, nullptr, {});
  NCCLTester::register_variants(library);

  auto store = runtime->create_store(lg::Shape{10000}, lg::int32());
  auto task = runtime->create_task(library, NCCLTester::TASK_CONFIG.task_id());
  task.add_output(store);
  task.add_communicator("nccl");
  runtime->submit(std::move(task));

  std::ignore = lg::finish();
}

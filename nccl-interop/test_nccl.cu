#include <iostream>
#include <legate.h>
#include <nccl.h>

#define CHECK_CUDA(...)                                                        \
  do {                                                                         \
    cudaError_t __result__ = (__VA_ARGS__);                                    \
    check_cuda(__result__, __FILE__, __LINE__);                                \
  } while (false)

void check_cuda(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    fprintf(stderr,
            "Internal CUDA failure with error %s (%s) in file %s at line %d\n",
            cudaGetErrorString(error), cudaGetErrorName(error), file, line);
    exit(error);
  }
}

#define CHECK_NCCL(...)                                                        \
  do {                                                                         \
    const ncclResult_t result = __VA_ARGS__;                                   \
    check_nccl(result, __FILE__, __LINE__);                                    \
  } while (false)

void check_nccl(ncclResult_t error, const char *file, int line) {
  if (error != ncclSuccess) {
    static_cast<void>(fprintf(
        stderr,
        "Internal NCCL failure with error %d (%s) in file %s at line %d\n",
        error, ncclGetErrorString(error), file, line));
    std::exit(error);
  }
}

namespace lg = legate;

struct NCCLTester : public lg::LegateTask<NCCLTester> {

  static inline const auto TASK_CONFIG = lg::TaskConfig{lg::LocalTaskID{0}};
  static inline const auto GPU_VARIANT_OPTIONS =
    lg::VariantOptions{}.with_communicators({"nccl"});

  static void gpu_variant(lg::TaskContext context)
  {
    // Communicators cached by runtime
    // reused when using the same devices
    auto comm = context.communicators().at(0).get<ncclComm_t*>();

    auto output = context.output(0);
    auto shape = output.shape<1>();
    auto volume = shape.volume();
    auto ptr = output.data().write_accessor<uint64_t, 1>().ptr(shape.lo);

    // Streams managed by runtime
    // for tying CUDA effects with task completion
    auto stream = context.get_task_stream();
    CHECK_NCCL(ncclAllReduce(ptr, ptr, volume, ncclUint64, ncclSum, *comm, stream));
  }
};

int main()
{
  lg::start();

  constexpr uint64_t size = 100;
  auto runtime = lg::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  const auto ngpus = machine.count(lg::mapping::TaskTarget::GPU);
  if (ngpus < 2) {
    std::cerr << "Need at least 2 GPUs" << std::endl;
    return 1;
  }
  if (size % ngpus != 0) {
    std::cerr << "Number of GPUs must be a multiple of " << size << std::endl;
    return 1;
  }

  auto library = runtime->create_library("test_nccl", lg::ResourceConfig{}, nullptr, {});
  NCCLTester::register_variants(library);

  auto store = runtime->create_store(lg::Shape{size}, lg::uint64());
  runtime->issue_fill(store, lg::Scalar{1ul});

  auto task = runtime->create_task(library, NCCLTester::TASK_CONFIG.task_id());
  task.add_input(store);
  task.add_output(store);
  runtime->submit(std::move(task));

  auto p_store = store.get_physical_store();
  auto acc = p_store.read_accessor<uint64_t, 1>();
  for (auto it = lg::PointInRectIterator<1>{p_store.shape<1>()}; it.valid(); ++it) {
    if (acc[*it] != ngpus) {
      std::cerr << "wrong contents" << std::endl;
      return 1;
    }
  }

  std::ignore = lg::finish();
  return 0;
}

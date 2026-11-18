#pragma once

// A few threads that share the loops of the host side of a multiplication: converting rows and weights to what
// the NPU reads and reading its results back. The thread that asks takes part in the loop, and nothing of the
// loop is left running when it returns. The threads are kept between loops: starting them for every
// multiplication would cost more than what they save.
//
// Built with OpenMP (XDNA_OPENMP, when ggml has it), the loops run on the threads of OpenMP instead: the same
// ones the CPU backend of ggml runs its operations on, which wait awake for a while after each of its parallel
// regions. The threads of the pool sleep between loops, and a loop of 0.2 ms waited for them to wake up while
// those of OpenMP took the same cores.

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class XdnaWorkers {
public:
  XdnaWorkers() = default;
  ~XdnaWorkers();
  XdnaWorkers(const XdnaWorkers &) = delete;
  XdnaWorkers & operator=(const XdnaWorkers &) = delete;

  // How many threads work at once, counting the one that asks: 1 or less leaves it alone. Threads are added when
  // more are asked for than there are; they are not removed when fewer are, they just stay out of the loops.
  void     mSetThreads(uint32_t pThreads);
  uint32_t mThreads() const { return vLimit; }

  // Runs pTask(0) ... pTask(pCount - 1), each once, in any order and several at once, and returns when all of
  // them have ended. One loop at a time: it is not to be called from two threads, nor from inside a task.
  void mForEach(uint32_t pCount, const std::function<void(uint32_t)> & pTask);

  // Runs pTask on the calling thread while, built with OpenMP, the other threads of its team wait asleep. After
  // every parallel region the threads of OpenMP spin for some milliseconds, and generating tokens the CPU backend
  // runs a region every half millisecond, so they never stop: seven cores spinning made the NPU 34 % slower
  // (measured on 2026/11/15, Qwen3.5 4B generating: 110 ms per token against 82). For waiting for the NPU.
  void mWhileParked(const std::function<void()> & pTask);

private:
  void mRun(uint32_t pIndex);  // what a thread of the pool does until the pool is destroyed
  void mWork();                // takes tasks of the loop that is running until none is left

  std::vector<std::thread>                 aThreads;
  std::mutex                               vMutex;
  std::condition_variable                  vWake;  // a loop has started, or the pool ends
  std::condition_variable                  vIdle;  // the last task of a loop has ended
  const std::function<void(uint32_t)> *    vTask  = nullptr;
  uint32_t                                 vCount = 0;  // tasks of the loop
  uint32_t                                 vNext  = 0;  // the next one to take
  uint32_t                                 vDone  = 0;  // the ones that have ended
  uint64_t                                 vLoop  = 0;  // number of the loop, for the threads to see a new one
  uint32_t                                 vLimit = 1;
  bool                                     vStop  = false;
};

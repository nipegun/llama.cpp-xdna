#include "xdna-workers.h"

#ifdef XDNA_OPENMP
#include <omp.h>
#endif

XdnaWorkers::~XdnaWorkers() {
  {
    std::lock_guard<std::mutex> vLock(vMutex);
    vStop = true;
  }
  vWake.notify_all();
  for (std::thread & vThread : aThreads) {
    vThread.join();
  }
}

void XdnaWorkers::mSetThreads(uint32_t pThreads) {
  const uint32_t vWanted = pThreads < 1 ? 1 : pThreads;
  {
    std::lock_guard<std::mutex> vLock(vMutex);
    vLimit = vWanted;
  }
#ifndef XDNA_OPENMP
  // the thread that asks is one of them
  while (aThreads.size() + 1 < vWanted) {
    aThreads.emplace_back(&XdnaWorkers::mRun, this, uint32_t(aThreads.size()));
  }
#endif
}

void XdnaWorkers::mForEach(uint32_t pCount, const std::function<void(uint32_t)> & pTask) {
  if (pCount == 0) {
    return;
  }
#ifdef XDNA_OPENMP
  if (pCount > 1 && vLimit > 1) {
    const int vThreads = int(vLimit);
    const int vCount   = int(pCount);
#pragma omp parallel for schedule(dynamic, 1) num_threads(vThreads)
    for (int i = 0; i < vCount; i++) {
      pTask(uint32_t(i));
    }
    return;
  }
#endif
  if (pCount == 1 || vLimit <= 1 || aThreads.empty()) {
    for (uint32_t i = 0; i < pCount; i++) {
      pTask(i);
    }
    return;
  }
  {
    std::lock_guard<std::mutex> vLock(vMutex);
    vTask  = &pTask;
    vCount = pCount;
    vNext  = 0;
    vDone  = 0;
    vLoop++;
  }
  vWake.notify_all();
  mWork();
  std::unique_lock<std::mutex> vLock(vMutex);
  vIdle.wait(vLock, [this] { return vDone == vCount; });
  vTask = nullptr;
}

void XdnaWorkers::mWhileParked(const std::function<void()> & pTask) {
#ifdef XDNA_OPENMP
  if (vLimit > 1) {
    // a region of the same size as the ones of the CPU backend, so the same threads: the first one runs the task
    // and the others sleep on a condition until it ends, instead of spinning
    std::mutex              vParkMutex;
    std::condition_variable vParkWake;
    bool                    vEnded   = false;
    const int               vThreads = int(vLimit);
#pragma omp parallel num_threads(vThreads)
    {
      if (omp_get_thread_num() == 0) {
        pTask();
        {
          std::lock_guard<std::mutex> vLock(vParkMutex);
          vEnded = true;
        }
        vParkWake.notify_all();
      } else {
        std::unique_lock<std::mutex> vLock(vParkMutex);
        vParkWake.wait(vLock, [&vEnded] { return vEnded; });
      }
    }
    return;
  }
#endif
  pTask();
}

void XdnaWorkers::mRun(uint32_t pIndex) {
  uint64_t vSeen = 0;
  for (;;) {
    {
      std::unique_lock<std::mutex> vLock(vMutex);
      vWake.wait(vLock, [&] { return vStop || vLoop != vSeen; });
      if (vStop) {
        return;
      }
      vSeen = vLoop;
      // the threads beyond the limit stay out
      if (pIndex + 1 >= vLimit) {
        continue;
      }
    }
    mWork();
  }
}

void XdnaWorkers::mWork() {
  for (;;) {
    const std::function<void(uint32_t)> * vFunction = nullptr;
    uint32_t                              vIndex    = 0;
    {
      std::lock_guard<std::mutex> vLock(vMutex);
      if (vTask == nullptr || vNext >= vCount) {
        return;
      }
      vFunction = vTask;
      vIndex    = vNext++;
    }
    (*vFunction)(vIndex);
    {
      std::lock_guard<std::mutex> vLock(vMutex);
      vDone++;
      if (vDone == vCount) {
        vIdle.notify_all();
      }
    }
  }
}

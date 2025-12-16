# tinyllava/utils/debug_watchdog.py
import os, sys, time, threading, signal, faulthandler, traceback

_last_tick = 0.0
_last_tag = "init"
_started = False

def touch(tag: str = "tick"):
    """在关键路径里调用，记录最近一次进度与标签"""
    global _last_tick, _last_tag
    _last_tick = time.time()
    _last_tag = tag

def _dump_stacks():
    """把所有线程堆栈打印到 stderr（不会退出进程）"""
    ts = time.strftime("%F %T")
    sys.stderr.write(f"\n================= STACK DUMP {ts} pid={os.getpid()} tag={_last_tag} =================\n")
    sys.stderr.flush()
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    sys.stderr.flush()
    sys.stderr.write("================= END STACK DUMP =================\n")
    sys.stderr.flush()

def _sigusr2_handler(signum, frame):
    _dump_stacks()

def install_sigusr2_dump():
    """允许外部 kill -USR2 <pid> 触发堆栈转储"""
    try:
        signal.signal(signal.SIGUSR2, _sigusr2_handler)
    except Exception as e:
        sys.stderr.write(f"[watchdog] WARN: install_sigusr2_dump failed: {e}\n")

def start_heartbeat(interval_sec: int = 15, log_file: str | None = None):
    """后台心跳线程：定期打印最近一次 touch 的时间与标签"""
    global _started
    if _started:
        return
    _started = True

    def _loop():
        while True:
            now = time.time()
            delta = (now - _last_tick) if _last_tick > 0 else -1
            msg = f"[HEARTBEAT pid={os.getpid()}] since_last={delta:.1f}s tag={_last_tag}\n"
            try:
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(msg)
                else:
                    sys.stderr.write(msg); sys.stderr.flush()
            except Exception:
                pass
            time.sleep(interval_sec)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()

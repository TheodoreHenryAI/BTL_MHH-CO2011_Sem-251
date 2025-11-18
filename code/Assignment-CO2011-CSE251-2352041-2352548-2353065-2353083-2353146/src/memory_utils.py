# src/memory_utils.py
from typing import Dict, Any

def _mb(x_bytes: int) -> float:
    return round(x_bytes / (1024.0 * 1024.0), 3)

def start_memory_tracking() -> Dict[str, Any]:
    tracker = {}
    # start tracemalloc if available
    try:
        import tracemalloc
        tracemalloc.start()
        tracker['tracemalloc'] = True
    except Exception:
        tracker['tracemalloc'] = False

    # attempt psutil for RSS
    try:
        import psutil
        p = psutil.Process()
        tracker['psutil'] = True
        tracker['rss_start'] = p.memory_info().rss
    except Exception:
        tracker['psutil'] = False
        tracker['rss_start'] = None

    return tracker

def stop_memory_tracking(tracker: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        'memory_tracked': False,
        'tracemalloc_current_mb': None,
        'tracemalloc_peak_mb': None,
        'rss_start_mb': None,
        'rss_final_mb': None
    }

    # obtain tracemalloc numbers if enabled
    if tracker.get('tracemalloc', False):
        try:
            import tracemalloc
            current, peak = tracemalloc.get_traced_memory()
            result['tracemalloc_current_mb'] = _mb(current)
            result['tracemalloc_peak_mb'] = _mb(peak)
            tracemalloc.stop()
            result['memory_tracked'] = True
        except Exception:
            pass

    # obtain rss via psutil if available
    if tracker.get('psutil', False):
        try:
            import psutil
            p = psutil.Process()
            rss_final = p.memory_info().rss
            rss_start = tracker.get('rss_start') or 0
            result['rss_start_mb'] = _mb(rss_start) if rss_start else None
            result['rss_final_mb'] = _mb(rss_final)
            result['memory_tracked'] = True
        except Exception:
            pass

    return result

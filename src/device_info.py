import json, os, platform, psutil, subprocess, shlex
import torch

# ---------- helpers ----------
def _run(cmd: str, timeout=4):
    try:
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL, timeout=timeout)
        return out.decode("utf-8", "ignore")
    except Exception:
        return ""

def _read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""

def _norm(s: str):
    if not s: return ""
    s = " ".join(s.replace("\x00", " ").split())
    bad = {"", "To Be Filled By O.E.M.", "INVALID", "Default string", "System Product Name", "None", "Unknown"}
    return "" if s in bad else s

def _join_unique(*parts):
    parts = [p for p in parts if p]
    # убираем дубли типа "Apple Apple M4 Max"
    res = []
    for p in parts:
        low = p.lower()
        if not res or low not in " ".join(res).lower():
            res.append(p)
    return " ".join(res)

# ---------- fastfetch (optional) ----------
def _from_fastfetch():
    out = _run("fastfetch --format json", timeout=2)
    if not out:
        return None
    try:
        data = json.loads(out)
    except Exception:
        return None

    cpu_name = ""
    gpu_name = ""
    gpu_cores = None
    for item in data:
        t = item.get("type","")
        if t == "CPU":
            r = item.get("result", {})
            name = _norm(r.get("cpu") or r.get("name") or "")
            # попробуем добавить физические ядра
            phys = r.get("cores", {}).get("physical")
            cpu_name = f"{name} ({phys})" if name and phys else name
        elif t == "GPU":
            r = item.get("result", [])
            if isinstance(r, list) and r:
                g = r[0]
                vendor = _norm(g.get("vendor",""))
                name = _norm(g.get("name",""))
                gpu_cores = g.get("coreCount")
                if vendor and name and not name.lower().startswith(vendor.lower()):
                    gpu_name = f"{vendor} {name}"
                else:
                    gpu_name = name or vendor
    if gpu_name and gpu_cores:
        gpu_name = f"{gpu_name} ({gpu_cores} cores)"
    return {"cpu": cpu_name or None, "gpu": gpu_name or None}

# ---------- macOS ----------
def _mac_host():
    sp = _run("system_profiler SPHardwareDataType")
    model_id = ""
    chip = ""
    for line in sp.splitlines():
        L = line.strip()
        if L.lower().startswith("model identifier:"):
            model_id = _norm(L.split(":",1)[1].strip())
        elif L.lower().startswith(("chip:", "processor name:")):
            chip = _norm(L.split(":",1)[1].strip())

    # Попробуем вытащить GPU ядра (не всегда доступно)
    dsp = _run("system_profiler SPDisplaysDataType")
    gpu_core_count = ""
    for line in dsp.splitlines():
        L = line.strip()
        if "Total Number of Cores" in L or "Количество ядер всего" in L:
            try:
                gpu_core_count = str(int("".join(ch for ch in L.split(":")[-1] if ch.isdigit())))
                break
            except Exception:
                pass

    # Красивое имя GPU
    gpu_name = chip if chip else "Apple Silicon GPU"
    if gpu_core_count:
        gpu_name = f"{gpu_name} ({gpu_core_count} cores)"

    # processor (CPU)
    cpu_line = ""
    for line in sp.splitlines():
        L = line.strip()
        if L.lower().startswith(("chip:", "processor name:")):
            cpu_line = _norm(L.split(":",1)[1].strip())
            break

    host = model_id or "Apple Mac"
    return host, cpu_line, gpu_name

# ---------- Linux ----------
def _linux_host():
    vendor = _norm(_read("/sys/class/dmi/id/sys_vendor") or _read("/sys/class/dmi/id/board_vendor"))
    product = _norm(_read("/sys/class/dmi/id/product_name"))
    version = _norm(_read("/sys/class/dmi/id/product_version"))
    if not (vendor or product):
        hc = _run("hostnamectl")
        maybe = []
        for line in hc.splitlines():
            if "Hardware Model" in line or "Chassis" in line:
                maybe.append(_norm(line.split(":",1)[-1]))
        host = _join_unique(*maybe) or "Linux Machine"
    else:
        host = _join_unique(vendor, product, version)

    # GPU через nvidia-smi
    gpu_name = ""
    gpu_mem_gb = None
    smi = _run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if smi:
        first = smi.splitlines()[0].split(",")
        if first:
            gpu_name = _norm(first[0].strip())
        if len(first) > 1:
            mem = "".join(ch for ch in first[1] if ch.isdigit())
            if mem:
                # memory.total в MiB
                gpu_mem_gb = round(int(mem)/1024, 2)

    # фолбэк ROCm
    if not gpu_name:
        rocm = _run("rocm-smi --showproductname --json")
        if rocm:
            try:
                j = json.loads(rocm)
                # очень разный формат у rocm-smi, возьмём что найдём
                if isinstance(j, dict):
                    for v in j.values():
                        if isinstance(v, dict) and "Card series" in v:
                            gpu_name = _norm(v["Card series"])
                            break
            except Exception:
                pass

    # CPU
    cpu_name = ""
    try:
        cpu_name = _norm(_run("lscpu"))
        if cpu_name:
            for line in cpu_name.splitlines():
                if line.lower().startswith("model name:"):
                    cpu_name = _norm(line.split(":",1)[1].strip())
                    break
    except Exception:
        pass
    if not cpu_name:
        cpu_name = platform.processor() or ""

    return host, cpu_name, gpu_name, gpu_mem_gb

# ---------- Windows ----------
def _win_host():
    name = _run('powershell -NoProfile -Command "Get-CimInstance Win32_ComputerSystemProduct | Select-Object -ExpandProperty Name"')
    host = _norm(name) or "Windows PC"
    cpu = _run('powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"')
    cpu = _norm(cpu.splitlines()[0] if cpu else "")
    # GPU
    gpu = _run('powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"')
    gpu = _norm(gpu.splitlines()[0] if gpu else "")
    return host, cpu, gpu

# ---------- public API ----------
def get_device_info():
    # Базовые поля
    info = {
        "platform": platform.system(),
        "device": "cpu",
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    # Пытаемся взять из fastfetch (опционально)
    ff = _from_fastfetch() or {}

    system = info["platform"]
    if system == "Darwin":
        host, cpu, gpu = _mac_host()
        info["processor"] = ff.get("cpu") or cpu or platform.processor()
        info["gpu_name"] = ff.get("gpu") or gpu or "Apple Silicon GPU"
        info["gpu_memory_gb"] = "shared with system RAM"
        info["host"] = host

    elif system == "Linux":
        host, cpu, gpu, gpu_mem = _linux_host()
        info["processor"] = ff.get("cpu") or cpu or platform.processor()
        info["gpu_name"] = ff.get("gpu") or gpu or None
        if gpu_mem:
            info["gpu_memory_gb"] = gpu_mem
        info["host"] = host

    elif system == "Windows":
        host, cpu, gpu = _win_host()
        info["processor"] = ff.get("cpu") or cpu or platform.processor()
        info["gpu_name"] = ff.get("gpu") or gpu or None
        info["host"] = host

    else:
        info["processor"] = ff.get("cpu") or platform.processor()
        info["host"] = f"{system} Machine"

    # ---- Torch backends (минимально инвазивно) ----
    # CUDA
    if torch.cuda.is_available():
        info["device"] = "cuda"
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_gb"] = round(total_mem / (1024**3), 2)
        except Exception:
            pass
        info["cuda_version"] = getattr(torch.version, "cuda", None)

    # MPS (если CUDA нет)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = "mps"
        # имя уже поставили выше для macOS; добавим shared-память, если не была задана
        info.setdefault("gpu_name", "Apple Silicon GPU")
        info.setdefault("gpu_memory_gb", "shared with system RAM")

    # ROCm (как подсказка; torch.version.hip доступен в сборках ROCm)
    hip_ver = getattr(torch.version, "hip", None)
    if hip_ver and info.get("device") == "cpu":
        # Если HIP есть, но torch.cuda недоступен, вероятно ROCm AMD
        info["device"] = "rocm"
        info.setdefault("gpu_name", "AMD GPU (ROCm)")

    # Финишная нормализация: не держим пустые поля
    for k, v in list(info.items()):
        if v in ("", None):
            del info[k]

    return info

# Пример:
if __name__ == "__main__":
    print(json.dumps(get_device_info(), ensure_ascii=False, indent=2))
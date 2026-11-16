#!/usr/bin/env python3
"""多连接分块下载器(curl 实现),针对 Zenodo 的单连接限速。

每个文件切成 N 块,N 个 curl 进程并行下载;每块可独立断点续传。
用法:
  python -I scripts/range_download.py <url> <output> [num_connections]

每块下载到 <output>.partXXXX,完成后拼接为 <output> 并校验大小。
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def resolve_url(url: str) -> tuple[str, int]:
    """跟随重定向,返回最终 URL 和文件大小。"""
    result = subprocess.run(
        ["curl", "-sIL", "--max-time", "60", url],
        capture_output=True, text=True, timeout=90,
    )
    final_url = url
    size = 0
    for block in result.stdout.split("\r\n\r\n"):
        headers = block.lower()
        if headers.startswith("http") and " 30" in block.split("\r\n")[0]:
            for line in block.split("\r\n"):
                if line.lower().startswith("location:"):
                    final_url = line.split(":", 1)[1].strip()
        if "content-length:" in headers and "content-range" not in headers:
            try:
                size = int(headers.split("content-length:")[1].split("\r\n")[0].strip())
            except ValueError:
                pass
    if not size:
        raise RuntimeError("无法获取文件大小")
    return final_url, size


def download_chunk(url: str, index: int, start: int, end: int, part: Path) -> None:
    """下载一个分块,支持从已有大小续传。"""
    while True:
        offset = part.stat().st_size if part.exists() else 0
        if offset >= end - start + 1:
            return
        result = subprocess.run(
            ["curl", "-sL", "--max-time", "900", "--speed-limit", "10240", "--speed-time", "60",
             "-r", f"{start + offset}-{end}", "-o", str(part), "-C", "-", url],
            timeout=960,
        )
        if result.returncode == 0 and part.exists() and part.stat().st_size >= end - start + 1:
            return
        time.sleep(min(30, 5 * (1 + result.returncode)))


def main() -> None:
    url, output = sys.argv[1], Path(sys.argv[2])
    connections = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    final_url, size = resolve_url(url)
    print(f"文件大小 {size / 1e9:.2f} GB,{connections} 连接")

    chunk_size = size // connections
    jobs = []
    start = 0
    for i in range(connections):
        end = size - 1 if i == connections - 1 else start + chunk_size - 1
        jobs.append((i, start, end, output.parent / f"{output.name}.part{i:04d}"))
        start = end + 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [
            pool.submit(download_chunk, final_url, i, s, e, p) for i, s, e, p in jobs
        ]
        for f in futures:
            f.result()

    # 拼接
    with open(output, "wb") as out:
        for i, s, e, p in jobs:
            with open(p, "rb") as chunk:
                while True:
                    data = chunk.read(1 << 22)
                    if not data:
                        break
                    out.write(data)
    actual = output.stat().st_size
    if actual != size:
        raise RuntimeError(f"大小不符:期望 {size},实际 {actual}")
    for _, _, _, p in jobs:
        p.unlink()
    elapsed = time.time() - t0
    print(f"完成 {output.name}:{size / 1e9:.2f} GB,用时 {elapsed / 60:.1f} 分钟"
          f"(平均 {size / 1e6 / elapsed:.1f} MB/s)")


if __name__ == "__main__":
    main()

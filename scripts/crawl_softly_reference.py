#!/usr/bin/env python3
"""Crawl softly.cc 中文页面,构建声音分类体系参考数据集。

范围与限制:
- 只抓页面文本、标题、描述和内嵌 YouTube 视频 ID(作为内容元数据)。
- 不下载、不存储任何音频文件;softly.cc 的音频无开放许可,不得用于训练。
- robots.txt 对所有爬虫放行;抓取间隔 0.6 秒,只抓 /zh/ 页面。

输出(Datasets/ 已在 .gitignore 中):
  Datasets/reference/softly_taxonomy/raw/*.html     原始 HTML,只读保留
  Datasets/reference/softly_taxonomy/softly_taxonomy.json  结构化分类数据
  Datasets/reference/softly_taxonomy/manifest.json  抓取清单与去重记录
"""

import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import requests

BASE = "https://softly.cc"
SITEMAP = "https://softly.cc/sitemap-0.xml"
DELAY_SECONDS = 0.6
USER_AGENT = "A2T-ResearchCrawler/0.1 (taxonomy reference; text-only, no audio)"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Datasets/reference/softly_taxonomy")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


class PageTextParser(HTMLParser):
    """收集标题、段落、列表项、链接和 YouTube 视频 ID。"""

    def __init__(self) -> None:
        super().__init__()
        self.headings: list[str] = []
        self.text_blocks: list[str] = []
        self.links: list[str] = []
        self.youtube_ids: set[str] = set()
        self._capture_tag: str | None = None
        self._capture_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._capture_tag = "skip"
            return
        attrs_dict = dict(attrs)
        if tag in ("h1", "h2", "h3", "h4"):
            self._capture_tag = tag
            self._capture_parts = []
        elif tag in ("p", "li"):
            self._capture_tag = tag
            self._capture_parts = []
        elif tag == "a":
            href = attrs_dict.get("href")
            if href:
                self.links.append(href)
        elif tag in ("iframe",):
            src = attrs_dict.get("src") or ""
            match = re.search(r"(?:embed/|vi/)([A-Za-z0-9_-]{11})", src)
            if match:
                self.youtube_ids.add(match.group(1))

    def handle_data(self, data: str) -> None:
        if self._capture_tag not in (None, "skip"):
            self._capture_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h1", "h2", "h3", "h4", "p", "li") and self._capture_tag == tag:
            text = re.sub(r"\s+", " ", " ".join(self._capture_parts)).strip()
            if len(text) >= 2:
                if tag.startswith("h"):
                    self.headings.append(text)
                else:
                    self.text_blocks.append(text)
            self._capture_tag = None
            self._capture_parts = []
        elif tag in ("script", "style") and self._capture_tag == "skip":
            self._capture_tag = None


def fetch(url: str) -> str:
    response = SESSION.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def select_urls(zh_urls: list[str]) -> tuple[list[str], dict[str, str]]:
    """挑选要抓的页面;时长变体只保留 1-hour 版,其余记为去重。"""
    keep: list[str] = []
    skipped: dict[str, str] = {}
    for url in sorted(set(zh_urls)):
        match = re.match(r".+/([a-z-]+)/(\d+-hours?)/?$", url)
        if match:
            category, duration = match.group(1), match.group(2)
            if duration != "1-hour":
                canonical = f"https://softly.cc/zh/{category}/1-hour/"
                skipped[url] = f"duration-duplicate:{canonical}"
                continue
        keep.append(url)
    return keep, skipped


def parse_page(url: str, html: str) -> dict:
    parser = PageTextParser()
    parser.feed(html)
    title_match = re.search(r"<title>([^<]*)</title>", html)
    desc_match = re.search(
        r'<meta (?:name="description"|property="og:description") content="([^"]*)"',
        html,
    )
    # 正文中的 YouTube ID(缩略图地址也可能出现)
    for video_id in re.findall(r"i\.ytimg\.com/vi/([A-Za-z0-9_-]{11})/", html):
        parser.youtube_ids.add(video_id)
    body = {
        "url": url,
        "title": title_match.group(1).strip() if title_match else "",
        "description": desc_match.group(1).strip() if desc_match else "",
        "headings": parser.headings,
        "text_blocks": parser.text_blocks,
        "youtube_video_ids": sorted(parser.youtube_ids),
        "internal_links": sorted(
            link for link in set(parser.links) if link.startswith("/")
        ),
    }
    return body


def main() -> None:
    raw_dir = OUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sitemap_xml = fetch(SITEMAP)
    all_urls = re.findall(r"<loc>([^<]+)</loc>", sitemap_xml)
    zh_urls = [
        u for u in all_urls
        if re.match(r"https://softly\.cc/zh(/|$)", u) and "zh-TW" not in u
    ]
    targets, skipped = select_urls(zh_urls)
    print(f"sitemap 共 {len(all_urls)} 个 URL,zh 页面 {len(zh_urls)} 个,去重后抓取 {len(targets)} 个")

    pages: list[dict] = []
    failures: list[dict] = []
    for index, url in enumerate(targets):
        slug = url.rstrip("/").split("/zh/")[-1].replace("/", "_") or "home"
        try:
            html = fetch(url)
            (raw_dir / f"{slug}.html").write_text(html, encoding="utf-8")
            page = parse_page(url, html)
            pages.append(page)
            if index % 20 == 0:
                print(f"[{index + 1}/{len(targets)}] {url}")
        except Exception as error:  # noqa: BLE001 — 单页失败不中断抓取
            failures.append({"url": url, "error": str(error)})
            print(f"FAILED {url}: {error}")
        time.sleep(DELAY_SECONDS)

    taxonomy = {
        "crawl": {
            "date": time.strftime("%Y-%m-%d"),
            "user_agent": USER_AGENT,
            "robots_policy": "softly.cc robots.txt allows all crawlers",
            "scope": "/zh/ pages only",
            "audio_policy": "text and embedded video IDs only; no audio downloaded "
            "(softly.cc audio has no open license, must not be used for training)",
        },
        "pages": pages,
    }
    manifest = {
        "sitemap": SITEMAP,
        "zh_urls_total": len(zh_urls),
        "crawled": len(pages),
        "failed": failures,
        "skipped_duration_duplicates": len(skipped),
        "skipped_map": skipped,
    }

    (OUT_DIR / "softly_taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成:成功 {len(pages)},失败 {len(failures)},时长变体去重 {len(skipped)}")
    print(f"输出目录:{OUT_DIR}")


if __name__ == "__main__":
    main()

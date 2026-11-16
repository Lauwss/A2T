# Softly 音频数据集构建规则(供本地 agent 执行)

版本:2026-11-17 | 适用:基于权利人授权的内部训练数据集,**不公开、不再分发**

## 1. 授权与边界(必须先读)

- 授权依据:`softly_permission_record.json`(项目所有者于 2026-11-17 与 Softly 交涉,允许爬取;数据集不得公开)。
- 数据集及其采集脚本全部放在 `Datasets/`(gitignored),**禁止**进入公开仓库(GitHub Lauwss/A2T 是公开的)。
- 禁止再分发任何原始音频或衍生切片;`commercial_use = false`,将来商用需与 Softly 单独签许可。
- 优先路径:如果 Softly 愿意直接提供原始音频文件,跳过 YouTube 抓取,直接用官方文件。

## 2. 需要携带到本地机器的文件(来自服务器)

| 文件 | 用途 |
| --- | --- |
| `Datasets/tools/fetch_softly_audio.py` | 抓取脚本(推荐直接用,别重写) |
| `Datasets/reference/softly_taxonomy/softly_video_index.json` | 75 个视频 ID 及分类、时长标签 |
| `Datasets/env_sounds/meta/softly_permission_record.json` | 授权记录 |
| `Datasets/tools/SOFTLY_DATASET_RULES.md` | 本文档 |

本地机器要求:可访问 YouTube(代理或海外机器);Python ≥3.10;`pip install yt-dlp`(≥2026.10);`ffmpeg`。

## 3. 数据范围(全套 12 类,75 条视频)

| Softly 分类 slug | A2T 类别 | 视频数 | 用途定位 |
| --- | --- | --- | --- |
| `rain-sounds` | `rain_stable` | 8 | v1 核心训练类 |
| `stream-sounds` | `water_flow` | 7 | v1 核心训练类 |
| `ocean-waves` | `ocean_waves` | 8 | 扩展登记 |
| `wind-sounds` | `wind` | 6 | 扩展登记 |
| `thunderstorm-sounds` | `thunderstorm` | 7 | 扩展登记(与 rain_stable 严格互斥) |
| `forest-sounds` | `forest_ambience` | 7 | 扩展登记 |
| `fireplace-sounds` | `fireplace` | 7 | 扩展登记 |
| `coffee-shop-sounds` | `coffee_shop` | 7 | 扩展登记 |
| `white-noise` | `white_noise` | 8 | 评估基线素材 |
| `pink-noise` | `pink_noise` | 6 | 评估基线素材 |
| `brown-noise` | `brown_noise` | 7 | 评估基线素材 |
| `lofi-music` | `lofi_music` | 6 | 登记类(音乐,不进环境音生成训练) |

说明:
- `guqin-ambient` 页面没有 YouTube 嵌入,本管线抓不到;如需古琴素材,人工确认该页音频来源后单独处理。
- 噪音三类(white/pink/brown)按 TAXONOMY 属于合成基线,这里的自然录音版本只作评估对照素材,不作生成训练素材。
- 视频清单以 `softly_video_index.json` 为准,共 **75 条**;同一视频跨分类复用时取索引里的第一个匹配分类。

体量估算:75 条 × 前 1 小时音频(m4a/opus 约 50–60 MB/条),合计约 **3–6 GB**,串行抓取时间主要取决于代理带宽(通常 1–3 小时)。

## 4. 抓取规格

- 工具:`yt-dlp`,每条视频只取**前 1 小时**音频:
  ```
  yt-dlp -f "bestaudio[ext=m4a]/bestaudio" \
    --download-sections "*00:00-01:00:00" --force-keyframes-at-cuts \
    --no-playlist --no-warnings \
    -o "raw/softly/<category>/%(id)s.%(ext)s" \
    "https://www.youtube.com/watch?v=<video_id>"
  ```
- 1 小时的依据:开头 15 分钟常带渐入、截断或非代表性段落;取满 1 小时原生动态,后续任何切片规格(8 秒训练窗、30–60 秒自然度检查、2–5 分钟长时测试)都从这一小时里选段切出,不再受源头限制。
- 限速:每条之间 `sleep 8`,串行执行,不开并行(尊重源站与 YouTube)。
- 失败重试:同一条最多 3 次;仍失败就记录跳过,不要卡死。

## 5. 落盘结构与命名

```
Datasets/env_sounds/
  raw/softly/<category>/<video_id>.m4a     原始格式,不转码
  meta/softly.json                          元数据(见 §6)
  meta/softly_permission_record.json        授权记录(随数据走)
```

## 6. 元数据 schema(每条记录)

```json
{
  "source_id": "softly_<video_id>",
  "platform": "softly_youtube",
  "platform_ref": "https://www.youtube.com/watch?v=<video_id>",
  "category": "rain_stable | water_flow",
  "license": "Softly-authorized-restricted",
  "attribution": "Softly (softly.cc)",
  "commercial_use": false,
  "durations_on_site": ["1-hour", "2-hours", "..."],
  "local_path": "...",
  "processing_version": "raw-v1",
  "quality_flags": {},
  "notes": "内部训练使用;数据集不公开;不得再分发"
}
```

顶层必须带 `"license_policy"` 与指向授权记录的说明。

## 7. QC 要求(抓完必做)

1. 每条 ffprobe:时长 ≥ 58 分钟(验证分节确实生效)、声道、采样率。
2. 解码检查:削波比例、静音比例、DC 偏移(可复用服务器上的 `scripts/inspect_env_sounds.py` 逻辑,元数据文件换成 softly.json)。
3. **长时真实性**:抽 2–3 条看频谱/自相关,确认是连续录音而非几十秒素材循环拼接;循环感明显的记录进 `quality_flags`。
4. 重复检测:75 个视频里同一视频可能跨分类复用,按 `video_id` 全局去重。

## 8. 完成判定与交付物

- **75/75** 下载成功(QC 通过)或失败清单完整、原因明确。
- `meta/softly.json` 生成,字段齐全,分类计数与 §3 表格对得上。
- 一份简短报告:成功数、失败原因、每分类总时长(分钟)、QC 问题清单(重点:长时真实性检查结论)。
- 把 `raw/softly/`、`meta/softly.json` 拷回服务器 `Datasets/env_sounds/` 对应位置。

## 9. 禁止事项

- 不把音频、切片、元数据或本规则文档推送到任何公开仓库、网盘或外部服务。
- 不在代码/日志里写入代理地址、密钥。
- 不下载超出 §3 范围和 15 分钟分节的内容。
- 不声称这些素材可商用;`commercial_use` 保持 false。

## 10. 已知风险与应对

| 风险 | 应对 |
| --- | --- |
| YouTube 下载签名/客户端失败 | 升级 yt-dlp 到最新版再试;仍失败记录跳过 |
| 视频实际是循环短素材 | QC 第 3 步发现后标记;循环素材对长时训练价值低,不强行使用 |
| Softly 撤回授权 | 立即停止,删除本地未入库素材,更新 permission record |
| 同一视频跨分类 | 以 `softly_video_index.json` 的 categories 字段为准,取第一个匹配分类 |

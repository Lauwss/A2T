# 环境声数据集状态报告

日期:2026-11-16
分支:`remote`

## 1. 已入库数据

| 来源 | 类别 | 片段数 | 大小 | 许可 |
| --- | --- | --- | --- | --- |
| FSD50K(经 hf-mirror 镜像) | `rain_stable` | 621 | — | CC0 / CC BY |
| FSD50K(经 hf-mirror 镜像) | `water_flow` | 1242 | — | CC0 / CC BY |
| **合计** | | **1863** | **2.24 GB** | 全部可商用 |

- 位置:`Datasets/env_sounds/raw/fsd50k/{rain_stable,water_flow}/`
- 元数据:`Datasets/env_sounds/meta/fsd50k.json`(逐条许可、来源、质检结果)
- 采样率/声道:全部 44.1 kHz 单声道
- 划分:`fsd50k dev` → train(1253 条),`fsd50k eval` → val(610 条),与原数据集划分一致

## 2. 质检结果

| 项目 | 结果 |
| --- | --- |
| 精确重复 | 0 条 |
| 削波 | 24 条(削波比 < 0.06%,可用但需标记) |
| 几乎静音 | 9 条 |
| 短于 10 秒 | 748 条 |

时长分布:

| 指标 | 值 |
| --- | --- |
| 最短 / 中位 / 均值 / 最长 | 0.31 s / 13.2 s / 13.6 s / 57.6 s |
| ≥ 8 秒 | 1260 条(68%) |
| ≥ 15 秒 | 821 条(44%) |
| ≥ 30 秒 | 8 条(0%) |
| 全库总时长 | 约 7.0 小时 |

## 3. 关键限制:不满足长录音要求

README §8.1 要求原始素材优先 1–10 分钟以上连续录音,长时测试需要原生 2–5 分钟。**FSD50K 最长只有 57.6 秒**,原因:

- FSD50K 的片段是 Freesound 上传者自行剪辑的短素材,不是连续长录音
- 这导致两个能力缺口:
  1. 无法学习长时间尺度的自然起伏(涨落周期、漂移)
  2. 无法做原生长时播放测试(只能拼接,而拼接结果按项目规范不得称为原生长时生成)

当前可用规模:**1260 条 ≥8 秒片段**,按 8 秒窗、4 秒步长估算约 3000–3800 个训练切片。足够跑通 MVP 和阶段 2(类别条件扩散),但不足以支撑正式训练规模。

## 4. 待补充

| 缺口 | 解决方案 | 阻塞原因 |
| --- | --- | --- |
| 1–10 分钟连续录音 | Freesound API(按许可过滤 + 时长过滤) | `api.freesound.org` 本机不通,需代理;需用户注册 API key |
| Wikimedia 雨声/水流 | 脚本已就绪 `scripts/download_wikimedia_sounds.py` | 所有 Wikipedia 域名本机不通,需代理 |
| 目标声(鼾声)库 | 待定来源 | 未开始 |

## 5. 已排除的来源

| 来源 | 排除原因 |
| --- | --- |
| Softly(softly.cc) | 音频经 YouTube 嵌入,无开放许可;仅页面文本作分类参考 |
| HF `igorriti/ambience-audio` | 只有 YouTube 元数据,无音频 |
| Zenodo 检索结果 | 无干净许可的雨声/水流长录音数据集 |
| FSD50K 的 CC BY-NC 片段 | 禁商用,已在清单阶段排除 |
| BBC Sound Effects | RemArc 许可禁商用 |

## 6. 复现命令

```bash
# 1. 分析 FSD50K 元数据生成子集清单(需 FSD50K.ground_truth.zip + FSD50K.metadata.zip)
#    产出 Datasets/env_sounds/meta/fsd50k_subset_plan.json

# 2. 按清单从镜像下载片段(断点续传,可重复运行)
/home/zmy/.conda/envs/bird/bin/python scripts/download_fsd50k_mirror.py 24

# 3. 质检(削波/静音/重复/时长),结果写回元数据
/home/zmy/.conda/envs/bird/bin/python scripts/inspect_env_sounds.py Datasets/env_sounds/meta/fsd50k.json
```

## 7. 数据使用注意

- 24 条削波片段:`quality_flags.has_clipping = true`,训练前决定是否剔除
- 9 条近静音片段建议剔除
- 短于 8 秒的 603 条不能做 8 秒切片,可用于短窗训练或直接排除
- 所有片段继承 FSD50K 原始 split,不得重新随机划分

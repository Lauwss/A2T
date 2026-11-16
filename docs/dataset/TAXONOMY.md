# A2T 数据集类别体系与元数据规范 v0.1

> 状态:草案。第一版范围以 README §8 为准:环境声为稳定雨声与连续水流,目标声为鼾声与人声。其余类别只登记、不采集,不进入第一版训练。

## 1. 设计来源

- README §8(数据层、元数据字段、划分规则)是本规范的基准。
- Softly 的分类体系作为命名与场景组织参考,数据在 `Datasets/reference/softly_taxonomy/`。
- 场景→推荐掩蔽声的先验映射在 `docs/dataset/softly_scenario_priors.json`。它是该站的编辑观点,**不是掩蔽效果的证据**,只用于生成掩蔽关系集时的初始候选排序。

## 2. 环境声类别(掩蔽声,masker)

| ID | 中文名 | 子类 | 版本状态 | 说明 |
| --- | --- | --- | --- | --- |
| `rain_stable` | 稳定雨声 | light_rain / steady_rain / heavy_rain | **第一版核心** | 排除雷声、明显风声干扰 |
| `water_flow` | 连续水流 | stream(溪流) / river(河流) / waterfall(瀑布) | **第一版核心** | 保留涨落,不人为削平动态 |
| `ocean_waves` | 海浪 | — | 扩展 | 需覆盖完整涨落周期,不沿用 8 秒规格 |
| `wind` | 风声 | — | 扩展 | — |
| `forest_ambience` | 森林环境声 | — | 扩展 | 含鸟鸣时需单独标注 |
| `fireplace` | 壁炉 | — | 扩展 | — |
| `coffee_shop` | 咖啡馆 | — | 扩展 | 含人声絮语,与目标声边界需说明 |
| `thunderstorm` | 雷暴 | — | 扩展 | 与 rain_stable 严格互斥,入库时检查 |

合成基线(只作评估对照,**不作为生成模型的训练素材**):

| ID | 说明 |
| --- | --- |
| `white_noise` | 白噪音 |
| `pink_noise` | 粉红噪音 |
| `brown_noise` | 棕噪音 |
| `spectral_shaped` | 频谱整形噪声,参数逐条记录 |

## 3. 目标声类别(被掩蔽声,target)

| ID | 中文名 | 子类 | 版本状态 | 说明 |
| --- | --- | --- | --- | --- |
| `snoring` | 鼾声 | continuous / intermittent | **第一版核心** | 保留事件、呼吸与安静间隔;记录事件起止与活动比例 |
| `speech` | 人声 | single_speaker / multi_speaker | **第一版核心** | 记录说话人匿名 ID、语音活动区间 |
| `footsteps_impact` | 楼板脚步/撞击 | — | 扩展 | 对应楼上邻居场景 |
| `street_traffic` | 街道交通 | — | 扩展 | — |
| `construction` | 施工 | — | 扩展 | — |
| `hvac` | 设备低鸣 | — | 扩展 | — |

Softly 场景页里的 `snoring-partner`、`upstairs-neighbours`、`street-noise`、`construction-noise`、`tinnitus` 等命名已并入上表。耳鸣自身不是环境噪声源,`tinnitus` 相关场景只作为掩蔽需求的参考,不设目标声类别。

## 4. 场景维度(掩蔽关系集的先验)

场景用统一 slug(`docs/dataset/softly_scenario_priors.json` 中的 79 个场景)。第一版只启用与 A2T 直接对应的场景:

| 场景 slug | 目标声先验 | 掩蔽声先验(来自 Softly,仅作候选假设) |
| --- | --- | --- |
| `snoring-partner` | snoring | pink_noise, brown_noise, rain_stable |
| `insomnia` | 无目标声(安静) | rain_stable, pink_noise, brown_noise |
| `noisy-neighbours` / `upstairs-neighbours` | footsteps_impact(扩展) | brown_noise, rain_stable |
| `street-noise` | street_traffic(扩展) | brown_noise, rain_stable, pink_noise |
| `tinnitus-masking` | tinnitus(非目标声库) | white_noise, pink_noise, rain_stable |

规则:先验只决定关系集构建时候选的**采样顺序**;最终关系以人工听评记录为准。频次统计(rain_stable 在 79 个场景出现 61 次)不能表述为「雨声掩蔽效果最好」。

## 5. 元数据 schema

### 5.1 环境声记录(每条原始素材)

```json
{
  "source_id": "wikimedia_0001",
  "platform": "wikimedia_commons | freesound | fsd50k | self_recorded | commercial",
  "platform_ref": "https://commons.wikimedia.org/wiki/File:...",
  "session_id": "wikimedia_0001",
  "category": "rain_stable | water_flow | ...",
  "subcategory": "stream | river | waterfall | null",
  "license": "CC0 | CC BY 4.0 | ...",
  "license_url": "...",
  "attribution": "作者与署名要求;CC BY 商用需保留",
  "commercial_use": true,
  "original_sample_rate": 48000,
  "channels": 2,
  "duration_s": 312.4,
  "crop_start_s": null,
  "crop_end_s": null,
  "processing_version": "raw-v1",
  "quality_flags": {"clipping": false, "silence_ratio": 0.02, "dc_offset": 0.001, "duplicate_of": null},
  "split": "train | val | test",
  "notes": ""
}
```

### 5.2 目标声记录

```json
{
  "source_id": "target_0001",
  "subject_id": "anon_001",
  "recording_id": "night_20260101_01",
  "category": "snoring | speech | ...",
  "device_environment": "录音设备与房间信息(已知时)",
  "privacy_record": "授权与隐私审查结论",
  "event_annotation_path": "events/target_0001.json",
  "duration_s": 30.0,
  "split": "train",
  "notes": ""
}
```

### 5.3 掩蔽关系记录

```json
{
  "relation_id": "rel_000001",
  "target_id": "target_0001",
  "masker_id": "wikimedia_0001",
  "level_basis": "window_rms | active_rms(必须固定并记录,不得混用)",
  "g_target_db": -6.0, "g_masker_db": 0.0, "g_common_db": -1.2,
  "mix_duration_s": 30.0,
  "alignment_offset_s": 0.0,
  "playback_settings": "",
  "eval_version": "peaq-fft-masking-mvp-v1",
  "proxy_scores": {"s_thr": null, "s_rep": null},
  "listening_record": null,
  "split": "train"
}
```

## 6. 目录结构

```
Datasets/
  env_sounds/
    raw/<platform>/<category>/...   原始素材,只读
    clips/                          训练切片(默认 8 秒),待处理层
    features/                       DAC 潜变量缓存,待处理层
    meta/<platform>.json            逐平台入库元数据
  target_sounds/                    待建
  relations/                        待建
  reference/softly_taxonomy/        分类参考(文本,无音频)
```

## 7. 划分规则(摘要,详见 README §9)

1. 先隔离来源,再切片:同一原始录音的裁剪不跨集合。
2. 目标声按个体(鼾声个体/说话人)隔离;环境声按原始录音与会话隔离;关系集两端都取训练集。
3. 比例约 80/10/10,服从独立来源数量,不按片段数机械分配。
4. 特征与切片继承原始 split,不重新随机划分。
5. 测试集不参与参数选择、候选规则与评分器训练。

## 8. 许可边界

- 只收 CC0 / Public domain / CC BY;CC BY-SA(传染)、CC BY-NC(禁商用)一律拒绝。
- CC BY 素材的 `attribution` 字段完整保留,商用署名清单由该字段直接生成。
- Softly 音频(YouTube 嵌入)无开放许可,禁止下载与训练使用;只有页面文本作为分类参考。
- 每个平台的入库元数据带 `license_policy` 声明与采集日期。

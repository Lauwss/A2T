# 阶段 1 验证报告:基础设施与冻结模型前端

日期:2026-11-17 | 分支:`remote`/`master`(已合并) | 环境:conda `bird`(Python 3.10.22、torch 2.11.0+cu128、6×RTX 3090)

## 1. 任务与依据

按 AGENTS.md 实施原则与 README §5.3 阶段 1 定义执行:**验证 BEATs 前处理、DAC encode/decode 往返、潜变量类型/形状/尺度/采样率,建立冻结 codec 的潜变量缓存**。本报告所有结论均为实测,未经实测的字段明确标注。

## 2. 环境修复记录(先于验证发生)

| 问题 | 处理 | 提交 |
| --- | --- | --- |
| `src/a2t/data/` 使用 `StrEnum`(需 Python ≥3.11),bird 是 3.10,主包无法导入 | 回移植为 `(str, Enum)` + `__str__` 覆写,行为等价;`pyproject` 对齐 `>=3.10`、`numpy>=1.24` | 0bfca7b |
| torchaudio 2.11.1 默认轮子链 CUDA 13 运行时,与 torch cu128 不配 | 改装 `torchaudio==2.11.0+cu128`(PyTorch 官方源) | 环境变更 |
| descript-audiotools 钉 `protobuf<3.20`,与 swanlab 要求 `>=6.31.1` 冲突 | 升级 protobuf 6.33.6;**已实测** audiotools 的 AudioSignal/DAC 路径正常;未使用的 tensorboard 写日志路径不受保护 | 已记录 |

测试基线:`pytest tests/` **63/63 通过**。

## 3. DAC 24kHz(S1-1/S1-2)

- 依赖:`descript-audio-codec 1.0.0`;检查点 `weights_24khz_8kbps_0.0.4.pth`(**MIT**,Descript),缓存于 `~/.cache/descript/dac/`
- 往返验证(`artifacts/dac_roundtrip/report.json`,6 条 ≥8s 样本,双类覆盖):
  - 长度:输入输出差 ≤8 样本(边缘效应)✓
  - 响度:Δ ≤ 0.1 LUFS ✓;峰值 0.78,无削波 ✓
  - LSD **7.5–8.2 dB**;对照 AAC 8kbps 同素材 LSD **37.6 dB**——谱重建显著更优
  - SI-SNR 在类噪声素材上为负(-26~+3 dB):GAN codec 相位不连贯的固有特性,**定性为不适验收指标**(有 AAC 对照佐证)
- **码率实测:24 kbps**(32 码本 × 10 bit × 75 fps 全部激活),发布元数据标 8kbps,**不符已记录**
- 听感抽检样本:`artifacts/dac_roundtrip/samples/`(本地,不入库)

## 4. BEATs 特征前端(S1-3)

- 代码:`microsoft/unilm` beats 子目录 vendored 至 `third_party/beats/`,**MIT**(LICENSE_UNILM.txt 已存档)
- 检查点:`BEATs_iter3.pt`(预训练版,AS2M,无分类标签;361.5 MB)经 hf-mirror 镜像 `lpepino/beats_ckpts` 获取,该镜像含 iter1/2/3 全系列可作后续消融
- 验证(`artifacts/beats_features/report.json`,GPU 0):
  - torch 2.11 加载需 `weights_only=False`
  - 输入规格:16 kHz 单声道、[-1,1](内部 ×2^15 做 kaldi fbank,128 mel)
  - 特征 `[B, patches, 768]`,实测 **~23.6 patches/秒**
  - 确定性 ✓、NaN free ✓
  - **padding mask:短样本在混长度批内特征渗漏(最大差 0.67;长样本决定批长度时无此问题)** → 对策:特征提取按长度分桶或逐条;阶段 2 训练用等长 8 秒窗天然规避

## 5. 潜变量缓存(S1-4)

- 脚本:`scripts/build_dac24_cache.py`(断点续传);产物 `Datasets/env_sounds/features/dac24/`(3.1 GB,float16)
- 覆盖:**1260 条 ≥8s**(rain 484 / water 776);跳过 603 条短片段;速度 4.6 条/秒
- 潜变量规范:hop=320 @24 kHz(**75 帧/秒**)、维度 1024、量化后连续向量 z(**非离散 code,禁止对 code 索引加噪**)
- 归一化规范(**仅 train split 拟合**,val/test 固定使用):

```
train: mean=0.0245  std=4.2986  absmax=25.94
val 参考:            mean=0.0285  std=4.3356(接近 → 划分无泄漏迹象)
```

## 6. swanlab(S1-5)

offline 模式全流程可用(roundtrip、cache、BEATs 三个 run 均已落盘 `swanlog/`);账号同步待 API key 配置后 `swanlab sync`。

## 7. 数据状态(输入侧)

- FSD50K 子集 1863 条(2.24 GB,CC0/CC-BY)入库,0 重复;QC:24 条削波、9 条近静音待剔除
- **缺口**:1–10 分钟连续录音(Softly 全套 75 条由本地 agent 抓取中,许可为权利人口头授权、数据集不公开);目标声库(鼾声/人声)未开始
- 完整状态见 `docs/dataset/DATA_STATUS.md`;Softly 授权边界见 `Datasets/env_sounds/meta/softly_permission_record.json`(私有区)

## 8. 风险与待验证

1. 听感抽检(往返样本)未做——指标合格不替代人耳确认
2. BEATs padding 渗漏已量化,但「分桶」策略对下游条件学习的影响待阶段 3 观测
3. DAC 潜变量归一化(absmax 25.9)需在扩散训练中配合缩放策略验证稳定性
4. GitHub 网络间歇不通,推送/克隆需择窗口;仓库尚未转私有,抓取脚本暴露中
5. 数据缺口(上节)是阶段 2→3 之间的最大进度风险

## 9. 下一步(阶段 2 入口)

用 `features/dac24/` + 类别标签训练类别条件扩散(遵循 README §5.3 阶段 2:先自然度后条件;所有训练/日志走 swanlab;仅 bird 环境;训练前确认 GPU 空闲,共享服务器上其他用户任务并行中)。

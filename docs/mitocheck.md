# DIAMOND 细胞轨迹基线

此版本保留 DIAMOND 的 EDM 去噪网络和采样器，用连续的 gene embedding 与时间条件替换 Atari 动作。默认输入一张 96×96 灰度细胞图，预测下一张；多步 rollout 将生成的图像作为下一步输入，在图像空间进行，不使用 latent rollout。该实验是条件细胞预测适配，非原论文 Atari 强化学习任务。

## 数据范围

使用 `splits/gene_track_split_v1/split_index.sqlite` 中 `pair_eligible=1` 的固定划分：6571 条训练轨迹、1642 条测试轨迹。实际图像从 `training_ready_v2/index.sqlite` 中读取，只接受 `paired_eligible=1` 且 `sample_type=single` 的细胞。排除额外轨迹、单帧轨迹及 composite 图像；不跨时间缺口构造序列，不跨母女细胞边界。保留原始划分，但它不能解释为完全未见的细胞家族。

默认从固定训练轨迹按 gene 分层、以 seed=0 确定性留出 5% 验证：6242 条用于优化、329 条用于验证、1642 条用于测试。history=1 时窗口数为 203068 / 10853 / 54049。测试集不用于选模型；best.pt 按验证损失保存。

图像 uint8 映射到 [-1,1]。条件包括 200 维 gene embedding、源帧绝对时间、目标帧绝对时间、源细胞年龄。默认时间为 `19 + frame_id × 0.5`，年龄为 `(frame_id - track_start_frame) × 0.5`。阴性对照依据 conditions 表映射到 NO_TARGET。未使用 siRNA embedding。与其他方法比较时请统一历史帧数、条件、时间、数据划分及测试窗口。

## 服务器安装和运行

上传修改后的整个 diamond 目录。保留 mitov2 的目录结构及图像符号链接指向的文件，服务器数据根目录应为 `/work/tl441/mitov2`。无需下载 Atari 数据或预训练 Atari 权重。

```bash
cd /path/to/diamond
conda create -n diamond-cells python=3.10 -y
conda activate diamond-cells
pip install -r requirements-cells.txt
export MITOV2_ROOT=/work/tl441/mitov2

# 先检查轨迹数、划分及部分图像，不开始训练
bash scripts/train_mitocheck.sh mode=audit output_dir=runs/cells-audit

# 默认 GPU 训练，100 epochs，batch_size=32
bash scripts/train_mitocheck.sh output_dir=runs/cells-seed0

# 继续训练：必须使用同一数据、模型和划分配置
bash scripts/train_mitocheck.sh resume=runs/cells-seed0/last.pt output_dir=runs/cells-seed0 training.epochs=200

# 完整测试：生成最多 10 步，保存前 16 组预测/真值图
python src/evaluate_mitocheck.py checkpoint=runs/cells-seed0/best.pt output_dir=runs/cells-seed0-test
```

显存不足可设置 `training.batch_size=8`。训练不依赖 wandb。输出包括 data_audit.json、metrics.jsonl、best.pt 和 last.pt。checkpoint 保存模型、优化器、随机状态、配置、轨迹成员和数据文件哈希；更换路径允许，修改数据或关键配置会拒绝恢复。改变 history、模型或条件后应开始新实验。

先进行服务器小规模试运行：

```bash
bash scripts/train_mitocheck.sh output_dir=runs/smoke training.epochs=1 training.batch_size=1 training.workers=0 training.max_batches=1 training.val_batches=1
python src/evaluate_mitocheck.py checkpoint=runs/smoke/best.pt output_dir=runs/smoke-test evaluation.max_windows=2 evaluation.horizon=2
```

试运行的截断设置只验证流程，不能用于正式比较。正式训练不要设置 max_batches/val_batches；正式测试不要设置 max_windows。CPU 可加 `device=cpu`。

## 评估解释

metrics.json 汇报 [0,1] 强度上的 MSE、MAE 和由汇总 MSE 计算的 PSNR，按 rollout 步数记录样本数，并按 gene 汇报单步结果。one_step 与 rollout_1 是同一次预测；后续步数反馈生成图像。每个合法起点都参与评估；轨迹末尾或缺口会截断，因此不同步数的样本数可能不同。长轨迹贡献更多窗口，比较实验也应使用相同加权方式。PNG 上排为预测，下排为真值，第一列为最后一帧输入。

本地已经验证 8 项测试、真实数据完整索引检查，以及一批 CPU 训练/验证和两窗口连续生成。尚未进行完整 GPU 训练，试运行损失和预测不能作为模型性能结论。

```bash
python -m unittest discover -s tests -p 'test_mitocheck.py' -v
```

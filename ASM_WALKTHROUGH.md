# 单色、单平面 ASM：第一次实际运行

本阶段只做固定理想传播模型下的连续相位优化，不连接硬件、不训练神经网络、
不使用多帧或滑动窗口。后续研究目标是类似 Holospeed 的滑动窗口多帧优化，
实验室设备为 4 bit 相位 PLM；这里的参数并不是该 PLM 的硬件标定。

## 1. 本机与独立环境

2026-10-16 检查结果：Intel MacBook Pro，四核 2 GHz i5，16 GB 内存，
Intel Iris Plus，macOS 14.8.2。已有 base Python 3.9.18 和 Cinnamoroll Python
3.11.11 环境均未安装 torch。使用后者的解释器创建本仓库 `.venv`，不向原环境安装包。

```sh
# 已执行；后续运行直接用 .venv/bin/python，不必 activate。
/Users/zack/opt/anaconda3/envs/Cinnamoroll/bin/python -m venv .venv
.venv/bin/python -m pip install -r requirements-asm.txt
.venv/bin/python -m pip check
```

使用 torch 2.2.2、torchvision 0.17.2、NumPy 1.26.4。完整依赖版本另存于
`requirements-asm-lock.txt`。这是本机小实验环境，不是整个训练/硬件功能的环境。
实测 `torch.cuda.is_available()` 和 `torch.backends.mps.is_available()` 均为 False。

为什么没有直接运行 README：

- `env.yml` 包含异常编码、Python 3.6、CUDA 11.3 和 Windows 绝对 prefix。
- `main.py` 内 `dev = torch.device('cuda')` 不适合本机。
- `main.py` 顶层导入硬件模块和 `prop_model.py`；后者依赖训练框架。
- 本阶段只需要 `prop_ideal.Propagation`、`algorithms.gradient_descent` 和相关工具。

因此增加 `run_asm.py` 作为学习入口，直接使用原始传播和优化函数。
原始 `main.py` 入口仍保留其原有依赖和 CUDA 假设。

## 2. 输入、尺寸和光学设置

默认图片为仓库已有 `data/test/rgb/0.png`（1920×1080 RGB）。另有
`frame_0043.png`（1024×436），本阶段只处理一张静态图。

| 参数 | 本实验值 | 含义 |
|---|---|---|
| device | CPU | 显式使用 CPU |
| channel | 1 | 取 RGB 图像的绿色通道，作为单色目标 |
| wavelength | 517.7 nm | `params.py` 中旧激光的绿色波长 |
| pixel pitch | 6.4 μm | `params.py` 中 LETO 像素间距 |
| propagation distance | 4.4 mm | VR 配置的中间平面（绿色索引 4） |
| F_aperture | 0.5 | 沿用仓库频域圆形孔径参数，不是 NA 的数值 |
| phase shape | 144×256 | 缩小后的相位网格，顺序是 H×W |
| loss ROI | 128×224 | 中心区域监督；边缘不直接计算 loss |
| initialization | uniform [-π, π), seed=0 | 相同种子用于复现 |
| optimizer | Adam, lr=0.03 | 更新的是相位张量 |

波长、像素间距、距离、孔径取自仓库的 VR 配置；网格、ROI、迭代数、学习率
是本次小实验设置。缩小网格而保持像素间距不变，会缩小模拟的物理窗口，
因此不能把这个结果当作原论文全分辨率复现。

默认使用单次 `Propagation(4.4 mm)`。原 `prop_model.model` 使用 `SerialProp`，
先传播到中间平面，再传播到所选目标平面；本次选择的就是中间平面，第二段距离为零。
本入口也没有引入其他深度平面。

## 3. 先小测试，再完整运行

在仓库根目录执行：

```sh
# 小测试：先验证梯度、loss 和保存/重载链路。
.venv/bin/python run_asm.py --iterations 30 --out results/asm_smoke

# 完整小实验：同一个种子，重新从相同初始相位开始。
.venv/bin/python run_asm.py --iterations 500 --out results/asm_cpu

# 记录与最佳相位对应关系的回归检查。
.venv/bin/python -m unittest test_asm -v
```

重复相同命令会覆盖相同输出目录的文件。若要保留另一组参数，使用不同 `--out`。
学习时可先用 `--iterations 5 --out results/asm_five_steps` 观察较短的过程。
如果更换颜色通道，需要一起指定对应的 `--wavelength-nm`。

本次实测（相同 seed，从相同初始相位分别运行）：

| 已评估迭代数 | 振幅 MSE |
|---|---|
| 1（初始相位） | 0.096483238 |
| 30 | 0.017019199 |
| 100 | 0.00408249 |
| 500 | 0.000941931 |

500 次运行的 loss 下降约 99.02%。方向导数检查相对误差约 0.00525%；
沿负梯度进行 RMS 为 0.001 rad 的微扰，loss 从 0.096483238 降到 0.096225068。
重载浮点相位并用新传播器重建，与保存时重建的最大差为 0。
两个回归测试通过，`pip check` 无依赖冲突。
预热后的 500 次实验计时约 2.25 秒（从首次传播至保存数值结果及重载检查，
不含 Python 模块导入、绘图和环境安装；不同机器/负载会有变化）。

## 4. 每一步的代码与物理过程

1. **目标图变为振幅**：`run_asm.load_target` 对应
   `image_loader.TargetLoader.load_image`。读取绿色通道，将 sRGB 灰度线性化为
   相对强度 `I_target`，再计算 `A_target = sqrt(I_target)`。
   保持纵横比中心裁剪，再缩小振幅图。目标不是复电场，只约束振幅。

2. **相位调制**：`prop_ideal.Propagation.forward` 中
   `torch.exp(1j * u_in)` 将实数相位变为 `u0 = exp(iφ)`。
   这是假定均匀单位振幅照明、仅调制相位的输入光场。

3. **自由空间传播**：`Propagation.compute_H` 构造
   `H(fx, fy) = aperture × exp(i z 2π sqrt(1/λ² - fx² - fy²))`。
   `Propagation.prop` 进行零填充、FFT、乘 H、逆 FFT 和裁剪。
   零填充至两倍尺寸用于减小 FFT 周期卷绕；频域各平面波分量获得不同传播相位，
   在输出平面干涉形成非均匀振幅。这里沿用原仓库的频率网格和孔径实现。

4. **测量模拟结果**：`algorithms.gradient_descent` 中
   `recon_field = forward_prop(slm_phase)`，随后取中心 ROI、计算
   `recon_amp = recon_field.abs()`。光强是 `recon_amp ** 2`。

5. **计算误差**：先求整体振幅缩放
   `s = mean(A_recon * A_target) / mean(A_recon ** 2)`，再计算
   `L = mean((s * A_recon - A_target) ** 2)`。
   `s` 是最小二乘意义下的整体亮度匹配，不是每个像素单独缩放。
   因为使用缩放和 ROI，这个 loss 不能直接代表衍射效率或绝对亮度。

6. **反向传播并更新相位**：`loss_val.backward()` 根据可微传播链条得到每个相位
   像素的 `∂L/∂φ`，`optimizer.step()` 用 Adam 的动量和自适应步长更新相位。
   一个像素的相位会通过衍射影响许多目标像素；这是全局耦合的优化。
   原仓库把 `s` 放在 `no_grad` 内；对这个最优最小二乘尺度，目标的相位梯度
   可以使用该尺度处的偏导。这里没有训练网络权重。

梯度下降的直观形式是 `φ ← φ - η∂L/∂φ`；实际执行的是 Adam，不能把每一步
精确当作这个简单公式。loss 不要求逐次严格下降，因此保存历史上最佳的已评估相位。

`run_asm.py` 在优化前额外做方向导数验证：比较 autograd 的预测和对相位进行
正负小扰动后的有限差分；并检查沿负梯度的小扰动使 loss 下降。

## 5. 输出怎么看

| 文件（`results/asm_cpu/` 内） | 内容 |
|---|---|
| `overview.png` | 目标、初始/优化重建、相位、loss 曲线、振幅误差 |
| `phase_radians.npy` | 连续浮点相位，单位 rad；真正用于再传播的结果 |
| `phase_preview.png` | 相位映射到 0–255 的非反相预览；不是 PLM 驱动文件 |
| `target.png`, `reconstruction.png` | 将目标/重建振幅平方，再 sRGB 编码的可视图 |
| `reconstruction_amplitude.npy` | 乘全局尺度 s 后的重建振幅 |
| `reconstruction_intensity.npy` | 上述振幅的平方，线性相对强度，未做显示裁剪 |
| `loss.csv` | 每次的振幅 MSE、尺度 s 和相位梯度 RMS |
| `metadata.json` | 参数、输入哈希、版本、耗时、梯度与重载检查结果 |

PNG 强度图用相同的 [0,1] 显示范围，高于 1 的值只在显示时裁剪。
科学数组没有这个显示裁剪。重建来自连续相位 `.npy`，并不是把 8 bit PNG
重新解码后得到的结果，更不是 4 bit 量化相位的结果。

## 6. 对原仓库做了哪些小修改

- `algorithms.py`：填充原先为空的 `loss_vals`；在 Adam 更新前对最佳相位做
  `detach().clone()`，同时保存对应振幅和 loss，避免引用随参数更新而改变，
  以及“更新前 loss、更新后相位”的错位。最佳初值改为无穷大。
  增加可选迭代回调，用于记录梯度大小和进度。
- 约定：第 t 次 loss 在第 t 次 Adam 更新前计算；返回所有已评估状态中的最佳相位。
  最后一步更新后的状态没有额外加入历史。`best_iteration=1` 表示初始相位。
- `prop_ideal.py`：显式写出原有 `meshgrid` 的 `indexing='ij'`，不改变其网格计算。
- 新增独立运行入口、依赖文件和两个实际 ASM 回归测试。

测试覆盖本阶段纯模拟路径。硬件、CITL、ADMM、神经网络训练不在此次验证范围内。

## 7. 后续如何衔接研究

先能解释 `loss.csv` 的下降以及相位/重建对应关系，再研究 16 级相位量化及
PLM 的真实相位映射。量化方法会影响可微优化，需单独设计和验证。
多帧平均通常涉及强度的时间累积；滑动窗口的具体目标、状态复用和帧约束，
应在核对 Holospeed 原文及硬件时序后另开实验，本次没有提前加入这些假设。

# 树莓派端无人机检测与多目标跟踪

基于 YOLOv5-Lite ONNX、ONNX Runtime、LK 光流和卡尔曼滤波的树莓派视觉跟踪项目。YOLO 检测负责提供目标框，多目标跟踪器负责跨帧关联和维护 ID；云台只跟随当前锁定的 active ID。


## 功能

- YOLO 检测框通过类别、尺寸和马氏距离门控后，使用两阶段匈牙利匹配与轨迹关联。
- 每条轨迹使用 6 维常速度卡尔曼状态 `[cx, cy, log(w), log(h), vx, vy]`。
- YOLO 更新中心和框尺寸；通过质量门控的 LK 光流仅更新速度观测。
- 检测结果携带源帧序号和时间戳；迟到检测在有界历史内回放到源时刻处理。
- 维护多条轨迹，但只允许一个 active ID 驱动云台；目标失效或选择有歧义时保持云台指令。
- 摄像头采集、YOLO 检测和主跟踪循环分工运行；实时摄像头默认只保留最新帧，减少排队延迟。
- CSV 和 JSON 汇总记录推理耗时、关联结果、轨迹状态、LK 质量、控制有效性和云台指令。

## 目录

```text
github_release/
├── active_track_frontend.py
├── async_detection.py
├── classes.names
├── dj_show2.py
├── latest_frame_capture.py
├── multi_target_tracker.py
├── pan_tilt_control.py
├── protected_template.py
├── roi_affine_tracker.py
├── target_appearance.py
├── target_selector.py
├── models/                 # 放置本地 ONNX 权重；权重不提交到 Git
└── ...                     # 其余融合和兼容模块
```

## 环境

- Raspberry Pi OS 64 位或其他支持 ONNX Runtime 的 Linux 环境。
- Python 3.9 或更高版本。
- USB 摄像头或可由 OpenCV 打开的摄像头。
- 如需云台控制：PCA9685 驱动板、舵机和 CircuitPython ServoKit。
- 一个与本项目预处理和输出格式兼容的 YOLOv5-Lite ONNX 模型。

安装推理和图像处理依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

树莓派系统提供的 OpenCV 包也可使用；如果没有 `cv2`，请在当前 Python 环境安装兼容的 OpenCV。`onnxruntime` 的可用安装包取决于操作系统、处理器架构和 Python 版本。

如需驱动 PCA9685 云台，另安装可选依赖：

```bash
python -m pip install -r requirements-servo.txt
```

## 模型文件

将 ONNX 模型放到 `models/best.onnx`。默认类别文件为 `classes.names`，其中类别顺序必须与模型一致；当前示例只写入一类 `UAV`，对应类别 ID 0。

也可以通过命令行指定模型和类别文件：

```bash
python3 dj_show2.py --modelpath /path/to/model.onnx --classfile /path/to/classes.names
```

ONNX 文件已被 `.gitignore` 排除，不会随 Git 提交。

## 运行

先只运行检测和跟踪，不控制舵机：

```bash
python3 dj_show2.py --source 0 --no-servo
```

确认摄像头画面、检测框、轨迹 ID 和日志正常后，运行云台控制：

```bash
python3 dj_show2.py --source 0
```

程序默认尝试初始化 PCA9685 舵机控制器。无云台或无控制板时使用 `--no-servo`。退出可按 `q` 或 Ctrl+C。

常用选项：

```bash
python3 dj_show2.py --help

# 保存带框视频
python3 dj_show2.py --source 0 --no-servo --save-video

# 显示关联候选、门控拒绝原因和检测源帧调试图
python3 dj_show2.py --source 0 --no-servo \
  --mot-show-all-tracks --mot-debug-detections \
  --mot-debug-association-matrix --save-video

# 处理本地视频；不使用舵机
python3 dj_show2.py --source input.mp4 --no-servo --no-display --no-video
```

多目标相关默认值包括：最大轨迹数 8、单次最多关联 16 个检测、强检测阈值 0.35、弱检测和新轨迹置信度下限 0.15、两个不同源帧确认轨迹、丢失后保留 3 秒、位置预测上限 0.35 秒、迟到结果历史窗口 0.5 秒。可用 `--help` 查看全部参数。

## 舵机参数

默认水平通道为 1、俯仰通道为 0，home 指令为 135°/90°，软件行程范围为 0°～270°，脉宽范围为 500～2500 微秒。这些值是当前项目硬件配置，换云台或支架后必须按实际接线、方向、行程和脉宽重新设置。

先用小步单轴测试确认接线和方向：

```bash
python3 dj_show2.py --servo-test pan
python3 dj_show2.py --servo-test tilt
```

`--home-only` 可只下发初始角度，不打开摄像头或加载模型。没有编码器或 IMU 时，程序记录的是舵机命令角度，不是实际机械角度。

## 日志与性能口径

每次运行会生成 `frames.csv` 和 `summary.json`。运行 FPS 是整条主循环处理帧率；YOLO 推理耗时只统计 ONNX Runtime 推理调用，两者含义不同。异步运行时，YOLO 检测频率低于主循环 FPS 属正常情况。

轨迹 ID 是图像时序关联 ID，不是无人机真实身份或 ReID 结果。没有带逐帧框和身份标注的视频，就不能用日志中的 loss 次数或 control position valid 比例代替检测召回率、MOTA 或 IDF1。

## 说明

- 默认多目标路径由本项目 `multi_target_tracker.py` 生成轨迹 ID，不依赖检测器内的 SORT ID。
- 根目录保留了部分历史兼容入口；本 README 的支持路径是默认 4.0.17 多目标模式。
- 模板、光流、恢复和云台相关阈值是工程默认值，需结合目标尺度、摄像头和标注视频验证。
- 代码中不包含模型训练、数据集、模型权重或树莓派硬件校准记录。

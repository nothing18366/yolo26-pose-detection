# yolo26-pose-detection

偏瘫手康复训练动作计数原型，当前版本 `1.0.0`。本项目不用于临床评估或诊断。

## 结构

```text
main.py              Qt 桌面入口
server.py            Android 使用的 FastAPI/WebRTC 服务
inference.py         Qt 与服务端共用的姿态和手部推理
actions/             六个动作状态机及公共上下文
metrics.py           关键点几何计算
tracking.py          跨帧手部跟踪
video_templates/     动作参数模板
models/              YOLO 与 MediaPipe 模型
tests/               回归测试
tools/               视频回放和模板工具
```

## 运行

安装桌面依赖后运行 `python main.py`。Android 服务依赖单独安装：

```bash
python -m pip install -r requirements-server.txt
python server.py --host 0.0.0.0 --port 8000
```

## 回归测试

```bash
python tests/test_rehab.py
python tests/test_inference.py
python tests/test_server.py
```

Android 协议和 Jetson 部署见 [服务端与 Android 联调](docs/服务端与Android联调.md)。动作规则、已验证结果和局限见[视频动作与阈值说明](docs/视频动作标准.md)。

## 工具

```bash
python tools/video_replay.py --video PATH --action fist --side left
python tools/video_template_tool.py --video PATH --action rub --output video_templates/rub_hand_back.json
```

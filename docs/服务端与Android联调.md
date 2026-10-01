# Android 与 Jetson 联调

## Python 服务

Android 客户端连接服务主机局域网地址，默认端口 `8000`。服务端需要 Python 依赖、`models/` 和 `video_templates/`：

```bash
python -m pip install -r requirements-server.txt
python server.py --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health
```

`GET /health` 返回服务状态、模型设备和版本。客户端向 `POST /rtc/offer` 发送 SDP offer、患者信息和动作 key，服务返回 SDP answer。服务支持单个活动训练客户端；第二个客户端收到 HTTP 409。WebRTC 使用 host ICE candidates，不配置 STUN/TURN。

DataChannel `rehab` 发送 `started`、`result`、`summary`、`error`；客户端发送 `stop` 结束并在收到汇总后发送 `ack`。视频轨上传服务端并回传标注画面。动作3自动识别背伸/掌屈，旧请求中的 `mode` 字段忽略。协议字段和动作 key 以服务端实现为准。

## Android 工程

当前客户端工程独立于本仓库。构建要求 Gradle JDK 11、AGP 7.1.2、Gradle 7.2、ButterKnife 10.2.1；WebRTC SDK 固定 `io.github.webrtc-sdk:android:114.5735.11`，不可直接升级到要求 Java 17 字节码的版本。工程使用 Android Studio 配置的 SDK 和 Gradle wrapper 构建、安装。

客户端默认采集 640×480、15 FPS，上传码率起始值 1.5 Mbps、上限 2.5 Mbps。显示镜像不改变上传算法帧方向；训练中切换本地/远端预览不会停止识别。服务端回传 H.264 初始码率为 2 Mbps，拥塞控制可下调。提高分辨率会增加推理、编码和传输开销；真机 FPS 需重新测量。

## Jetson Orin Nano SUPER 8GB

先在设备核实 JetPack、Python、NVIDIA PyTorch/CUDA、MediaPipe、PyAV 和 aiortc 版本。先安装与 JetPack 匹配的 PyTorch，再安装服务端依赖；将 `models/` 和 `video_templates/` 一同部署。MediaPipe 及 ARM64 PyAV wheel 的兼容性、H.264 编码负载和端到端帧率尚未在 Jetson 真机验收。

## 联调验收

- 摄像头方向正确，服务端标注视频持续返回，连续运行时处理/回传 FPS、RTT、跳帧和丢包可观察且延迟不持续增长。
- 六个动作都能开始、结束并收到结果和汇总；相同输入在 Qt 与服务端的动作分项结果一致。
- 服务断开、App 切后台和第二客户端接入时会话能正确释放或拒绝。
- 视觉事件数需与人工逐次标注比较；模型推理 FPS、模拟 WebRTC 测试和程序计数都不能代替真人准确率验收。

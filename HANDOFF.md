# Handoff

## 项目

偏瘫手康复训练动作计数原型，版本 `1.0.0`，不是临床评估或诊断工具。主仓库入口为 Qt `main.py` 和 Android WebRTC 服务 `server.py`，二者共用 `inference.py`、`actions/`、`metrics.py` 和 `tracking.py`。Android 工程单独位于其自己的仓库。

## 当前实现

- 六个动作及顺序由 `actions.REGISTRY` 集中管理；key 为 `rub`、`fist`、`stretch`、`cup`、`abduction`、`press`。动作参数在同名 JSON 模板中。
- 动作3自动识别背伸/掌屈；动作4统计身体相对腕部抬起和回落，不依赖水杯检测；动作5分别统计拇指和食指伸出回位。
- Android 协议为 FastAPI + 单客户端 WebRTC，DataChannel 名称为 `rehab`。结果类型包括 `started`、`result`、`summary`、`error`；客户端通过 `stop` 结束、以 `ack` 确认汇总。
- 模型文件位于 `models/`。YOLO 与 MediaPipe 的具体模型权重商业分发许可尚未确认，不要添加未经核实的许可证声明。

## 验证与限制

- 回归命令：`python tests/test_rehab.py`、`python tests/test_inference.py`、`python tests/test_server.py`。
- 动作视频回放和程序输出、人工复核区间记录在 `docs/视频动作标准.md`。除明确标注外，程序计数不是人工真值或准确率。
- 尚未完成多患者、多机位逐次人工对照；Jetson ARM64 依赖、真实设备帧率和模型商业许可仍待核实。
- 长时间遮挡、透视和相机移动会造成漏计或误计；视觉代理不能判定疼痛、施力、抓握安全或医学合格。

## 部署

依赖和 Android/Jetson 联调步骤见 `docs/服务端与Android联调.md`。服务端安装入口为 `requirements-server.txt`，启动入口为 `python server.py --host 0.0.0.0 --port 8000`。更改服务端代码后需重启进程。

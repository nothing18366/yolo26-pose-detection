# 康复动作识别服务

偏瘫手康复训练辅助原型，服务版本 1.2.0。提供六动作识别，以及 Android 使用的账号、患者、计划、教学、记录和 FastAPI/WebRTC 服务。不用于临床评估或诊断。

Android 1.3.0 支持两种日常模式：本地骨架模式只回传关键点，远端标注模式回传服务端绘制的视频。工作人员在设备设置中选择，下次进入训练生效，患者同样适用；另外保留一次性远端原图诊断。采集仍请求 640×480、15 FPS，六动作模型与计数规则保持原有版本。

推理引擎保留可选的标注绘制功能，供需要标注图像的工具调用。

## 启动

在原推理环境、项目根目录运行：

```bash
python -m pip install -r requirements-server.txt
python server.py --host 0.0.0.0 --port 8000
```

模型后台加载，业务接口可先使用；GET /health 的 model_loaded=true 才能训练。服务只允许一个活动训练客户端。

## 结构与文档

应用代码在 src/rehab_service，模型在 models，动作参数在 video_templates，公共教学在 media；data 为本地数据库，不提交 Git。

- [结构与命名](docs/architecture.md)
- [开发、测试与工具](docs/development.md)
- [HTTP/WebRTC 协议](docs/api.md)
- [部署、SMTP 与备份](docs/deployment.md)
- [训练画面、骨架协议与诊断](docs/display_modes.md)
- [性能与验证结论](docs/performance.md)
- [动作规则、验证结果和局限](docs/action_rules.md)
- [尚未完成与实机验收](docs/remaining_work.md)

Android 工程独立维护，两端更新时需核对协议兼容。模型权重商业分发许可仍待确认。

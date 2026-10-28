# 康复动作识别服务

偏瘫手康复训练辅助原型，服务版本 1.1.0。提供六动作识别、Qt 桌面端，以及 Android 使用的账号、患者、计划、教学、记录和 FastAPI/WebRTC 服务。不用于临床评估或诊断。

Android 日常训练使用本地视频与服务端骨架数据，普通模式没有回传视频轨。工作人员可仅为下一次训练开启远端原图诊断。采集仍请求 640×480、15 FPS，六动作模型与计数规则保持原有版本。

## 启动

在原推理环境、项目根目录运行：

```bash
python -m pip install -r requirements-server.txt
python server.py --host 0.0.0.0 --port 8000
```

Qt 桌面端安装相应依赖后运行 `python main.py`。模型后台加载，业务接口可先使用；GET /health 的 model_loaded=true 才能训练。服务只允许一个活动训练客户端。

## 结构与文档

应用代码在 src/rehab_service，模型在 models，动作参数在 video_templates，公共教学在 media；data 为本地数据库，不提交 Git。

- [结构与命名](docs/architecture.md)
- [开发、测试与工具](docs/development.md)
- [HTTP/WebRTC 协议](docs/api.md)
- [部署、SMTP 与备份](docs/deployment.md)
- [本地骨架、诊断与回滚](docs/local_skeleton.md)
- [640×480 性能对照结果](docs/performance_640x480.md)
- [动作规则、验证结果和局限](docs/action_rules.md)
- [尚未完成与实机验收](docs/remaining_work.md)

Android 工程独立维护，两端更新时需核对协议兼容。模型权重商业分发许可仍待确认。

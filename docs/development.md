# 开发与验证

Python 3.10+，使用原推理环境。服务依赖在 requirements-server.txt。可选 `python -m pip install --no-deps -e .` 安装源码包，根入口与测试脚本也会定位 src/。本项目按完整源码目录部署，模型和媒体不包含在 Python wheel 中。

```bash
python server.py --host 0.0.0.0 --port 8000
```

## 回归

```bash
python tests/test_actions.py
python tests/test_inference.py
python tests/test_skeleton.py
python tests/test_rtc.py
python tests/test_business.py
python tests/test_auth_schedule_feedback.py
python tests/test_patient_password.py
python tests/test_business_rtc.py
python tests/test_teaching.py
python tests/test_structure.py
```

业务/RTC 测试使用临时数据库和邮件传输替身，不连接正式患者库、不发送真实邮件。test_rtc.py 与 test_business_rtc.py 使用真实 ICE/视频/DataChannel 和假推理，验证协议、记录、占用、重复结束与令牌撤销。

## 工具

```bash
python tools/video_replay.py --video /路径/视频.mp4 --action fist --side left
python tools/video_template_tool.py --video /路径/视频.mp4 --action rub --output video_templates/rub_hand_back.json
python tools/backup_records.py /备份目录/rehab.sqlite3
```

回放默认样本改用仓库内教学视频；完整参考视频请通过 --video 指定。回归通过不能代替实际患者、Jetson、摄像头和 SMTP 验收。

## 固定输入与机器人采样

固定教学视频与时间戳用于核对算法输入、阶段、计数和合格结果；开关绘制时应保持一致：

```bash
python tools/benchmark_videos.py artifacts/current/videos-before --no-draw
python tools/benchmark_videos.py artifacts/current/videos-after
python tools/compare_benchmarks.py artifacts/current
```

机器人只连接独立夹具：

```bash
REHAB_PERFORMANCE_DIR=artifacts/current/server python tests/rtc_fixture.py
python tools/benchmark_robot.py artifacts/current/robot --device 机器人IP:端口 --server http://服务端IP:18121 --modes skeleton annotated --repeats 1
python tools/compare_benchmarks.py artifacts/current
```

夹具固定 18121 端口，强制使用临时数据库并清除 SMTP 配置；不是正式部署入口。`--test-return-pause` 验证 annotated 断流回退与手动恢复，暂停/恢复接口只存在于测试夹具。每段默认 15 秒预热＋30 秒采样，`--sample-seconds 15` 可缩短；工具保留未完成采样/异常，不把外部强停后的完整数据当成 JUnit 通过。正式测量前备份机器人连接与登录配置，结束后恢复。

汇总工具读取 `robot/` 与 `server/`，输出 summary.json/metrics.csv；比较固定输入时读取 videos-before/videos-after。历史阶段目录和旧视频回传模式已移除。素材标记工具使用 OpenCV 桌面 GUI，需另备有 GUI 支持的开发环境；日常服务继续使用 headless 依赖。

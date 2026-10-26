# 开发与验证

Python 3.10+，使用原推理环境。服务依赖在 requirements-server.txt；Qt 入口还需要 PySide6 和 OpenCV 图形支持。可选 `python -m pip install --no-deps -e .` 安装源码包，根入口与测试脚本也会定位 src/。本项目按完整源码目录部署，模型和媒体不包含在 Python wheel 中。

```bash
python main.py
python server.py --host 0.0.0.0 --port 8000
```

## 回归

```bash
python tests/test_actions_ui.py
python tests/test_inference.py
python tests/test_rtc.py
python tests/test_business.py
python tests/test_auth_schedule_feedback.py
python tests/test_patient_password.py
python tests/test_business_rtc.py
python tests/test_teaching.py
python tests/test_structure.py
```

无显示设备运行 Qt 测试时设置 QT_QPA_PLATFORM=offscreen。业务/RTC 测试使用临时数据库和邮件传输替身，不连接正式患者库、不发送真实邮件。test_rtc.py 与 test_business_rtc.py 使用真实 ICE/视频/DataChannel 和假推理，验证协议、记录、占用、重复结束与令牌撤销。

## 工具

```bash
python tools/video_replay.py --video /路径/视频.mp4 --action fist --side left
python tools/video_template_tool.py --video /路径/视频.mp4 --action rub --output video_templates/rub_hand_back.json
python tools/backup_records.py /备份目录/rehab.sqlite3
```

回放默认样本改用仓库内教学视频；完整参考视频请通过 --video 指定。回归通过不能代替实际患者、Jetson、摄像头和 SMTP 验收。

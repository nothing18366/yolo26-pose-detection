# 服务部署与数据维护

适用于可信机构局域网、单进程、单活动训练客户端。Android 与服务端需要配套更新；匿名 RTC 请求不受支持。

## 启动

在项目根目录及原推理环境执行：

```bash
python -m pip install -r requirements-server.txt
python server.py --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health
```

部署需包含 `src/`、入口文件、`models/`、`video_templates/` 和完整 `media/`。路径以项目位置解析，不依赖启动工作目录。服务版本为 1.2.0，数据库 schema 保持 3。

Android 使用主机可互访的局域网地址；模拟器访问本机通常为 `http://10.0.2.2:8000`。模型后台加载，`model_loaded=true` 才可识别；模型失败时认证、患者、计划和记录仍可用。`busy=true` 表示已有活动客户端。代码或教学素材修改后需要重启服务。不要增加 Uvicorn worker 数量共享当前单设备推理状态。

## SMTP 与账号

`.env.example` 仅作字段模板，程序不自动加载 `.env`。由 shell 或部署工具导入环境：

```bash
export SMTP_HOST='邮件服务器'
export SMTP_PORT='587'
export SMTP_SECURITY='starttls'
export SMTP_FROM='发件邮箱'
export SMTP_USER='发件账号'
export SMTP_PASSWORD='邮箱授权码'
```

SSL 使用 `SMTP_SECURITY=ssl`，通常端口 465；两种模式都校验证书。SMTP_USER 省略时使用发件邮箱。未配置或发送失败返回 503，没有固定码或控制台码后备入口。

初始工作人员账号为 `admin` / `123`。患者账号由工作人员分配，可不绑定邮箱；邮箱用于验证码登录和密码找回。账号、权限及训练流程见 Android 仓库 `docs/usage.md`，接口见 [api.md](api.md)。验证码与令牌仅存摘要；备份也包含这些敏感认证数据。

## 数据与备份

默认数据库 `data/rehab.sqlite3`；环境变量 `REHAB_DB` 可指定持久目录，更换路径不会自动迁移数据。v2/v3 迁移代码必须保留：已有库升级前自动建立一致性备份，患者和历史 ID 不变，不自动生成患者密码。

运行中使用 SQLite backup API，包含已提交 WAL 数据：

```bash
python tools/backup_records.py /备份目录/rehab.sqlite3
python tools/backup_records.py /备份目录/rehab.sqlite3 --database /数据目录/rehab.sqlite3
```

已有备份不会覆盖。恢复前停服务、备份当前目录，把备份放入新目录并用 REHAB_DB 指向它；不混入旧 WAL/SHM。重启将备份中的活动训练标记中断。不要在运行时只复制主数据库而遗漏 WAL。

## 教学素材

`media/` 包含六组 mp4、jpg、srt，必须一起部署。字幕为教学引导，不是音轨逐字转写。重启服务计算文件摘要与内容版本；旧审核只属于旧版本，新内容自动待审核，试用中待审核状态可见但不自动阻止训练。

重新生成视频与封面需 FFmpeg，日常服务播放现有素材不需要：

```bash
python tools/prepare_teaching_media.py /路径/合集.mp4
```

同步 Android 内置教学文字可增加 `--android-assets /Android工程/app/src/main/assets`。

## Jetson 与网络

Jetson Orin Nano SUPER 上先核实 JetPack、Python、CUDA 与 NVIDIA PyTorch 版本，再安装业务依赖，避免普通 CPU PyTorch 覆盖 CUDA 版本。推荐系统 Python 配合 venv。当前固定的 MediaPipe 0.10.35 缺少官方 PyPI Linux ARM64 wheel，需核实厂商包或自行构建；当前依赖不能据此视为可直接在 Jetson 安装。

还需实机验证 PyAV/aiortc、H.264 编解码负载、可写空间和 SMTP 出站。WebRTC 只使用 host ICE candidates，不配置 STUN/TURN，要求两端网络可互通。

Android 采集请求 640×480、15 FPS，实际相机可能更快；上传起始码率 1.5 Mbps、上限 2.5 Mbps。RK3588 保留软件 VP8，其他设备保留 H.264。本地骨架模式只上传视频，返回关键点与业务数据；远端标注模式回传绘制后的画面，工作人员诊断回传未标注原图。只有远端标注执行服务端绘制；两种回传模式均沿用已有编码兼容策略，H.264 初始目标 2 Mbps，拥塞控制可下调。

更新后重启服务，并确认 `/health.capabilities.local_skeleton_v1=true` 和 `annotated_video_v1=true`；旧进程不会自动加载新源码。新 App 在开始训练前按所选模式检查能力，不支持时提示升级服务端。工作人员诊断见 [display_modes.md](display_modes.md)。

## 现场验收

- 实体设备检查摄像头方向、中文 TTS、前后台退出及断线释放。
- 六动作各至少 30 秒，记录处理 FPS、编码耗时、延迟、丢帧和热状态；与同设备同输入基线比较。
- 逐次人工对照真实动作，不能用程序输出或模拟器 FPS 充当准确率。
- 邮件必须完成真实收信和登录；备份恢复必须在独立目录演练。

## 版本与回滚

源码版本使用 Git 提交定位；APK 保存版本号、versionCode、SHA-256 与签名证书摘要。更新前结束训练、备份业务库，并保留一套配套服务与同证书 APK。正式库、签名、本机配置与构建产物不提交 Git。

回滚服务时使用独立目录检出对应提交，复用已核对的配置与资源；不要用旧数据库覆盖新增训练记录。APK 覆盖安装需同证书且 versionCode 不低于已安装版本，不通过卸载绕过而丢失本地数据。曾安装 versionCode 5 回滚包的设备，后续升级需构建更高 code。两端均需检查 health、画面、计数与保存。

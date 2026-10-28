# HTTP 与 WebRTC 协议

除登录接口、`/health` 和公共 `/media/...` 外，请求统一带 `Authorization: Bearer <token>`。JSON 错误格式为 `{"detail":"中文原因"}`。401 表示未登录/令牌失效，403 表示角色或归属不允许，409 表示状态冲突，429 表示限流，503 表示服务条件未就绪。接口完整字段可在 `/docs` 查看。

| 方法与路径 | 内容与权限 |
| --- | --- |
| POST `/api/v1/auth/email-code` | `{email}`；真实发送，成功返回 sent、retry_after、expires_in，不包含验证码 |
| POST `/api/v1/auth/email-login` | `{email,code}`；返回 token、expires、expires_at、role、patient_id |
| POST `/api/v1/auth/staff-login` | `{username,password}`；角色由服务端返回 |
| POST `/api/v1/auth/patient-login` | `{username,password}`；只允许患者账号，返回同一登录令牌结构 |
| POST `/api/v1/auth/password-reset-code` | `{username,email}`；匹配绑定后发送真实重置邮件，不返回验证码 |
| POST `/api/v1/auth/password-reset` | `{username,email,code,password}`；设置患者新密码并撤销旧登录 |
| GET/PUT `/api/v1/patients/{id}/account` | 仅工作人员；GET 返回 username/password_configured，PUT 接收 username 和可选 password（首次必填） |
| GET `/api/v1/auth/me` | id、role、patient_id、expires |
| POST `/api/v1/auth/logout` | 撤销当前令牌 |
| GET/POST `/api/v1/patients` | 工作人员搜索/建档；建档字段 name、affected_side、forearm_cm、可空 email |
| GET/PATCH `/api/v1/patients/{id}` | 本人或工作人员读取；仅工作人员修改，email 为 null/空串表示解绑 |
| GET/PUT `/api/v1/patients/{id}/plan` | 读取当前和 pending；仅工作人员保存 `{items:[{action,target_reps}],weekdays:[1,3,5],effective_from:"YYYY-MM-DD"}` |
| GET `/api/v1/patients/{id}/calendar?week_start=YYYY-MM-DD` | 所在周七天、训练日、完成/达标、due_days、scheduled_days、completed_days、adherence；仅本人/工作人员 |
| GET `/api/v1/patients/{id}/overview` | 本人或工作人员读取当前计划、今日训练及完成情况；日期固定按北京时间计算 |
| GET `/api/v1/patients/{id}/trends?action=cup&start=YYYY-MM-DD&end=YYYY-MM-DD` | 可选 mode=plan/single、revision；最多 366 天；服务端线程聚合 |
| POST `/api/v1/training-runs` | `{patient_id,mode:"plan"}` 或 single + action + target_reps；患者额外受到生效计划限制 |
| GET `/api/v1/training-runs?patient_id=...&days=0&offset=0` | 当前患者记录，最多 50 条；days=0/7/30 |
| GET `/api/v1/training-runs/{id}` | 不可变患者/动作/条件/教学版本快照、sessions、feedback、end_reason |
| POST `/api/v1/training-runs/{id}/finish` | 结束尚未完成整份计划，不覆盖已有结果 |
| PUT `/api/v1/training-runs/{id}/feedback/pre` 或 `/post` | `{state:"submitted",pain:null,fatigue:0,discomfort:false,note:null,difficulty:null}` 或 `{state:"skipped"}`；同阶段更新不重复插入 |
| POST `/api/v1/sessions` | `{run_id,item_index}`；按计划顺序，必须先保存或跳过前测 |
| GET `/api/v1/sessions/{id}` | 最终结果丢失时查询，不能继续旧计数 |
| POST `/api/v1/sessions/{id}/interrupt` | `{}` 为 client_exit；也支持 `{reason:"user_discomfort"}`，重复调用不重复结算 |
| GET `/api/v1/actions` | 六动作说明、封面、版本、files 各文件 sha256/bytes/url、review |
| PATCH `/api/v1/actions/{key}/review` | 工作人员提交 `{content_version,state:"pending/approved/rejected",reviewer,note}`；版本不符返回 409 |

`POST /rtc/offer` 必须带同一 Bearer 令牌及 `{type:"offer",sdp:"...",session_id:"...",preview_mode:"keypoints"}`。`preview_mode` 默认 keypoints，仅上传视频；diagnostic 仅允许工作人员，并协商回传解码原图视频，权限检查先于占用训练名额。服务读取会话快照，不信任 offer 里的患者或次数覆盖。第二客户端返回 409。可靠 `rehab` 通道保持 started/result/summary/error 字段，附带 session_id；发送 `{type:"stop"}` 正常结束，或 `{type:"stop",reason:"user_discomfort"}`。可靠保存、重复结束、ack 丢失释放和最新帧队列仍沿用现有实现。

`GET /health` 增加 capabilities：local_skeleton_v1、clock_sync_v1、diagnostic_video、diagnostic_png。`GET /rtc/sessions/{id}/frame.png` 仅工作人员、仅当前诊断会话可取当前已解码帧，响应 image/png、Cache-Control: no-store，不在服务端落盘；无当前诊断图像返回 409。骨架与时钟消息见 [local_skeleton.md](local_skeleton.md)。


## 训练与结果语义

- 动作 key 固定为 `rub/fist/stretch/cup/abduction/press`；业务 DataChannel 为可靠 `rehab`，骨架使用独立无序、零重传的 `pose`。
- 会话状态为 `pending/connecting/running/completed/interrupted`；整次训练状态为 `active/completed/unfinished/interrupted`。`completed` 表示正常结束，是否达到目标通过 `target_reached` 判断。
- 准备页只开本地预览；倒计时完成后建立 RTC，首帧处理时开始计时。中断后不能继续旧计数，新训练建立新会话。
- `elapsed_seconds` 是首帧至结束的真实经过时间，休息不计入；`summary.total` 是已识别动作累计时长，`summary.avg` 是单次动作平均时长。完成为零时页面显示暂无判定。
- `summary` 先保存数据库再发给客户端；重复 stop 返回同一结果，ack 丢失时自动释放名额。丢失汇总可通过会话查询接口恢复。
- 每秒最多保存一次检查点；服务重启将遗留活动训练标记中断，最后检查点不能解释成完整结果。
- 患者与计划快照不会被后续编辑回写。训练中修改对应患者、计划或账号返回 409。

## 数据与坐标约定

JSON 使用 `snake_case`。`patient_id` 是患者标识，`run_id` 是整次训练，`session_id` 是一个动作会话；不可混用。业务日期按北京时间计算，跨午夜按整次训练创建日期归属；时间戳带时区。

模型内部 `pts` 为 MediaPipe 顺序的 21×2 图像像素坐标，`world` 为可选 21×3 模型手部世界坐标；人体姿态使用 COCO-17。左右侧表示人体左右，显示镜像不改变上传帧。像素、比例、角度、厘米和秒保持各自单位。

趋势按动作、目标、患侧、前臂长度、计划版本、训练类型及算法指纹分组，组间不连线。正常结束的结果用总合格次数/总完成次数汇总；部分结果单列，旧记录条件不完整时不补造。

`GET /api/v1/training-runs` 的 days 支持 0/7/30，offset 分页，每页最多 50 条；趋势区间最多 366 天。`GET /api/v1/patients/{id}/overview` 的当前实现固定按北京时间汇总，保留的 utc_offset_minutes 参数不改变该行为。

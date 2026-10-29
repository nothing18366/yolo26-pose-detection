# 训练画面与诊断

配套服务端 1.2.0、Android 1.3.0。工作人员在“设置 → 设备与服务”选择日常模式，默认本地骨架；选择保存在本机，退出登录保留，患者登录沿用。进入计划或单项训练时固定选择，计划内各动作沿用；修改在下一次进入训练时生效。

| 模式 | 训练页切换 | 服务端 | Android |
| --- | --- | --- | --- |
| keypoints | 本地／骨架 | 推理、业务结果、pose 消息；不绘制或回传视频 | 本地预览与骨架覆盖层 |
| annotated | 本地／远端标注 | 推理、绘制、编码回传与业务结果；不发送 pose | 本地预览或远端播放器，不绘制本地骨架 |
| diagnostic | 本地／骨架＋工作人员原图入口 | 推理、pose、未标注视频与按需 PNG | 本地骨架及独立诊断播放器 |

准备阶段显示本地。识别开始后默认骨架或远端标注；远端首帧前显示本地并提示等待，首帧到达后自动切换。连续 3 秒无新远端帧后由周期检查退回本地，恢复后手动切回。显示切换不重连、不重置计数或计时；annotated 切回本地仍继续回传和解码，降低开销需下一次训练选择 keypoints。

“下一次训练开启远端诊断”仅工作人员可用，覆盖下一个成功连接的动作会话，不同时生成标注回传视频。成功连接后消耗开关，失败保留；下一动作恢复计划固定的日常模式。退出登录或切换服务清除诊断开关，保留日常模式。

患者只能访问本人会话的日常标注视频；diagnostic 和 PNG 由服务端检查工作人员权限，检查先于占用训练名额。PNG 只在当前诊断会话按需编码，不落盘，能帮助区分上传失真和回传压缩。当前会话不新增视频协商。

## 显示与识别边界

两种骨架显示共用 `display_hands`：真实检测并完成左右手分配后的平滑前坐标，复制后用于显示；桥接/预测手点不画成新检测。动作计数、几何与跟踪仍使用原有平滑数据，显示不回写算法。人体保持每两帧刷新一次。需要刷新人体时，全图手检测与人体检测并行，补检和动作更新仍使用当前帧人体结果；外展的手腕裁剪保留顺序执行。

本地视频不等待识别，不预测或补帧，骨架可能滞后。视频与骨架共用等比例留白、旋转和镜像；连续 300 ms 无可用更新即清空，恢复后重新显示。拒绝错会话、旧序号、无效坐标及已校准后的过时结果。结果中断 3 秒提示、10 秒中断及可靠保存流程保持不变。

三路径共用一次推理、一次动作更新和同一保存流程。只有 annotated 调用绘制并回传画布；diagnostic 回传原始解码帧。回传保留输入尺寸、PTS/time_base，使用最新帧队列；RK3588 使用软件 VP8，其他设备沿用 H.264。

## 骨架协议与计时

keypoints/diagnostic 创建独立 `pose` DataChannel，`ordered=false, maxRetransmits=0`。发送缓冲未清空时跳过更新；可靠 `rehab` 通道继续承载 started/result/summary、stop/ack。annotated 不创建 pose 通道，服务端也不发送坐标。

| 字段 | 定义 |
| --- | --- |
| type/version/session_id/seq | pose、1、会话、递增结果序号；丢弃更新可导致序号不连续 |
| width/height/rotation | 解码后正向图像尺寸，rotation=0 |
| decoded_at_ms/completed_at_ms | 服务端取出解码帧、生成结果的单调时钟毫秒 |
| body/body_state | COCO-17 `[x,y,confidence]`，无效点 null；fresh/cached/missing |
| hands | 左右侧、state、predicted、MediaPipe-21 `[x,y]`；非 fresh 或 predicted 时 points=null |
| tracking_confidence | 原有跟踪状态指标，不是医学置信度 |

坐标归一化至 0–1，按人体左右命名。手模型不输出逐点分数，hands.confidence=null。`clock_ping`/`clock_pong` 通过可靠通道估算时钟偏移，优先较小 RTT 样本；无法校准时仍按本地接收超时清除。

“解码→首次绘制”只估算服务端取出解码帧至 Android Canvas 首次绘制的耗时，不含采集、上传及之前的排队，不等于物理屏幕延迟。校准不可用显示不可用，误差约为校准 RTT 的一半。RTT 不等于画面延迟，各分段分位数不能直接相加。

点击训练页 FPS 可查看人体、左右手 fresh 更新率与最近帧耗时。`result.timing` 包含 `input_wait_ms/image_conversion_ms/worker_wait_ms/inference_ms/result_ms`；pose 另含 `result_to_send_ms/send_started_at_ms`。Android `skeleton.pipeline` 按 seq 记录 `parse_ms/ui_queue_ms/render_wait_ms/decoded_to_first_draw_ms/return_estimate_ms` 及对应服务端分段，未绘制消息不计首次绘制样本。

`pose_inference_ms` 在人体复用时为 0；`hand_detection_ms` 含转换、检测及补检；`tracking_ms` 含左右手分配、平滑和几何。人体与手检测可重叠。`hand_full_detections/hand_crop_detections` 为调用次数。更新率不含缓存、桥接、预测点；这些是诊断数据，不是动作准确率。

可选 `REHAB_PERFORMANCE_DIR` 导出服务端逐帧 JSONL，默认关闭；关键点及采样不写患者数据库，schema 保持 3。测试方法见 [development.md](development.md)，已有实测及限制见 [performance.md](performance.md)。

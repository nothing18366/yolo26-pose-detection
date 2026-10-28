# 结构与命名规范

## 模块职责

```text
src/rehab_service/
  app.py                 FastAPI 组装、启动、健康检查
  settings.py            项目资源目录
  api/                   认证、患者、训练、教学和 RTC 路由
  services/              认证规则、教学版本、动作会话、骨架消息序列化
  storage/               SQLite 访问及兼容迁移
  recognition/           共享推理、几何、手部跟踪、模板读取
    actions/             每个动作一个状态机，__init__.py 注册
  desktop/app.py         Qt 界面与采集线程
models/                  原始模型资产
video_templates/         动作参数 JSON
media/                   公共教学素材
tests/                   回归与联调测试
tools/                   回放、素材生成、数据库备份
```

根目录 main.py/server.py 只保留启动入口。HTTP 路由校验权限和参数后调用服务或存储；SQLite 事务保留在存储层。RTC 管理单客户端、视频队列和会话释放，调用共享推理引擎及 TrainingSession；动作状态机不依赖 HTTP、摄像头或数据库。

services/auth.py 与 storage/database.py 的现有调用方式保留，未增加 Repository 接口或空层。Qt 与服务端复用同一 PoseEngine 和六动作注册表。

## 命名

Python 包/模块、函数、变量使用 snake_case，类使用 UpperCamelCase，常量和环境变量使用 UPPER_SNAKE_CASE。单位写入名称；新配置使用 REHAB_ 前缀，现有 REHAB_DB/SMTP_* 保持兼容。测试按职责命名 test_*.py，不使用阶段名称。

JSON 字段、动作 key、数据库表/字段、命令行参数和环境变量属于兼容接口，不能随内部类名一起替换。模型权重目录与接口传输结构应分清。

## 维护规则

- 新动作增加独立模块、注册项、模板及对应回归。
- 改动路径时同步检查启动脚本、导入、测试替身、工具、资源定位和算法指纹。
- 算法指纹包含识别代码和模型。目录迁移和已验证的绘制开关仅对精确匹配的源码哈希保留原指纹，避免拆分历史趋势；后续识别代码或权重变化仍产生新版本。
- 数据迁移及异常/事务处理是兼容逻辑，不因文件名带旧版本而删除。
- 当前行为放入主题文档；未完成事项集中在 remaining_work.md，不保留相互覆盖的阶段说明。
- 患者数据库、令牌、密钥、本机配置、日志、缓存和构建产物不提交 Git。模型许可未核实前不添加许可证承诺。

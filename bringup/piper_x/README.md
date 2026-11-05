# Piper X 官方真机启动入口

当前电脑从任意目录打开终端后运行（迁移后按实际位置调整 `~/orbbec`）：

```bash
cd ~/orbbec
bash bringup/piper_x/start_real_control.sh
```

启动前按手册第0节检查CAN。接口缺失或未启用时，本入口会提示并退出，不自动配置CAN。

本入口会启动连接真机的官方驱动、MoveIt和RViz。启动参数与迁移前一致：`auto_enable=false`、5%速度、官方控制门控、ROS domain 93及单实例锁。只有现场准备完成后才人工启动。

完整采图、识别、规划和执行步骤见 [真机启动手册](../../graspnet_orbbec/docs/REAL_GRASP_STARTUP.md)。视觉和执行适配仍使用原有程序；不包含旧自研控制后端或仿真入口。

2026-11-04从“视觉”目录迁入；旧启动脚本只转发到本入口。新ROS日志与缓存写入本目录的`runtime/`，不纳入Git；原历史日志未移动。

迁移到其他电脑时的目录布局、环境与重新编译步骤见 [迁移路径说明](../../graspnet_orbbec/docs/PORTABLE_DEPLOYMENT.md)。

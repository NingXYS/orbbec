# 真机工程的路径与迁移约定

`orbbec/` 与 `PIPER X/` 保持同级，父目录和用户名可以变化。正式启动脚本按自身位置解析官方工作空间；标定绑定以 `graspnet_orbbec` 根目录解析 `../../PIPER X/hand-eye-calibration`。相对命令行参数仍以当前工作目录为准，不要求从旧路径启动。

每个操作终端先进入新的 `orbbec` 根目录，再执行：

```bash
orbbec_root="$(pwd -P)"
```

具体真机步骤见 [启动手册](REAL_GRASP_STARTUP.md)。正式入口与抓取适配分别是 `bringup/piper_x/start_real_control.sh` 和 `graspnet_orbbec/scripts/run_moveit_wire_grasp.sh`。

## 环境与编译

视觉环境为 `graspnet_orbbec/.conda`。标定机器人和相机入口优先使用 `PIPER X/hand-eye-calibration/.conda-robot`、`.conda-camera`；未建立工程内环境时，兼容当前用户的 `$HOME/miniconda3/envs/handeye-robot`、`handeye-camera`。其他位置用 `HANDEYE_ROBOT_CONDA_PREFIX`、`HANDEYE_CONDA_PREFIX` 指定。变量中的相对路径以 `hand-eye-calibration` 根目录解析。

例如，若环境位于 `PIPER X/robot-env`：

```bash
export HANDEYE_ROBOT_CONDA_PREFIX=../robot-env
```

`/opt/ros/humble`、系统Python、USB/CAN设备名等系统位置保持不变，需要在目标电脑安装或配置。ROS安装产物、Conda环境和Python/CUDA扩展不保证直接复制可用，应在目标机器重建。

标定相机程序可从两目录的共同父目录构建（路径以CMake源码目录为基准解析默认SDK）：

```bash
cmake -S "PIPER X/hand-eye-calibration/cpp" -B "PIPER X/hand-eye-calibration/build"
cmake --build "PIPER X/hand-eye-calibration/build" -j2
ctest --test-dir "PIPER X/hand-eye-calibration/build" --output-on-failure
```

生成程序的SDK运行搜索路径采用 `$ORIGIN` 相对路径。旧构建缓存仍包含旧机器路径，换电脑后先移出旧 `build` 目录再构建；官方ROS工作空间的 `build_humble`、`install_humble` 也需重建，不能手工批量替换生成文件。

## 历史记录

标定冻结报告、原始照片、旧执行报告和来源记录中的绝对路径保留作为采集时记录，不批量改写；冻结文件有SHA验证。它们不等同于当前依赖位置。仿真及其指向“视觉”的历史配置未纳入本次真机路径修改。

迁移后须重新核对设备、安装关系、桌面平面和当前反馈。离线测试及 `--help` 不证明真机取流、规划或实际抓取成功。

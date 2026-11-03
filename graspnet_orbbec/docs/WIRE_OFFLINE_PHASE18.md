# Piper X Phase1.8 离线识别、标注和避让复核

只读历史图像或指定图片，不打开相机/CAN、不启动实机驱动。Phase 2 BLOCKED，名义141.5mm工具未完成物理标定。新默认14mm是相对于原始交界的总偏移；22mm是名义接触带，完整夹爪模型另画红色外廓。

## 打开两张历史图

在项目Python环境中运行，每次使用新的输出目录：

```bash
python scripts/run_wire_offline.py --snapshot enabled --gui --out /tmp/wire_enabled_new
python scripts/run_wire_offline.py --snapshot raised --gui --out /tmp/wire_raised_new
```

`configs/wire_offline_snapshots.json`指向本机Phase1.8已绑定哈希的scene；没有文件时明确失败，不退回实时相机。独立任意图片可用`--image /绝对路径/image.png --recognize`，没有可靠几何时只保留像素和待定位状态。

## 窗口中做什么

银色线头是金属端子，灰色部分是绝缘线，红/绿块是支撑物。先按Z放大；仅核对橙色轮廓是否沿可见银色金属外缘。

| 操作 | 作用 |
|---|---|
| Shift+左键 | 在左原图确认金属/绝缘原始交界，重算14mm；旧轮廓确认失效 |
| Z | 4倍原图像素放大，点击自动映回原图，可细选轮廓 |
| T / 左键 / 右键 | 切到端子多边形编辑；左键加点、右键撤点 |
| Enter | 闭合当前多边形；尚未人工确认 |
| X | 清除轮廓和确认 |
| Y | 操作者确认轮廓及端子/绝缘关系；不代表抓取安全 |
| S | 保存不可覆盖的标注JSON、独立SHA256、识别结果和画布 |
| R | 重跑原AutomaticWireRecognizer；重用记录ROI，轮廓确认失效 |
| Q | 退出并导出当前结果；未按Y的草稿仍未确认 |

图例：绿标记为14mm抓取中心，黄色箭头为绝缘方向，紫色是22mm接触带，红色是完整夹爪模型的保守投影，橙色为独立轮廓证据。显示人工来源/确认状态、目标有效性和全部阻塞原因。3mm仅理想直线名义值；当前模型估计与保留的历史+0.991/−2.867mm分别显示，不强改为3mm。

## 回放、重跑、保存和重载

不加`--gui`就导出文件；加`--recognize`实际运行原本地SAM2/端子模型，默认历史回放只重用冻结识别。历史ROI来自原识别记录，分割算法不变，不下载/安装依赖。

```bash
python scripts/run_wire_offline.py --snapshot enabled --recognize --boundary-point 400 29 --out /tmp/wire_recognized_new
python scripts/run_wire_offline.py --snapshot enabled --annotation /路径/端子标注.json --out /tmp/wire_reload_new
python scripts/run_wire_offline.py --snapshot enabled --compare-10mm --out /tmp/wire_compare_new
```

新入口始终使用14mm。`--compare-10mm`只额外导出同锚点旧策略JSON，无执行授权。标注加载要求JSON与`.json.sha256`同时存在，并核对原图文件/解码像素、掩码、实例、原始交界及载荷哈希；跨图、改点、重分割后不能静默沿用确认。

原`scripts/run_wire_segmentation.py`也可加`--offline-scene scene.json --terminal-annotation annotation.json`进入同一计算/显示链。原窗口的普通相机分支不是本轮获准入口，本轮只用文件分支。

## 有效性与物理边界

人工轮廓保存在manual_terminal_annotations，terminal_detections仍是原自动模型输出，不能因人工标注变为自动成功。助手草稿ASSISTANT_VISUAL_DRAFT和测试数据SYNTHETIC_TEST不能自动变成人工确认。

同一端子因反光分成多片时，保留每片原始轮廓，并额外用整体凸包保守覆盖片间未知区域；高亮断口不能当作自由空间。当前两图已由用户在本轮对话明确复核，来源单独记为HUMAN_CHAT_REVIEW并绑定实际回复、原图与标注哈希；确认范围限于可见位置和同一端子身份，不包含尺度、隐藏形状或物理安全。v2标注存visible parts与avoidance_geometry两个层次；v1文件仍可载入，但避让时也采用保守外包络。

缺轮廓、尺度或几何时PREVIEW_ONLY；已知包络投影相交或绝缘覆盖不足可REJECTED，具体原因可见。完整模型将所有高度层保守投影，二维相交不等于已经证明实体3D碰撞。模型/42mm历史高度未验收时，毫米数字均标“模型估计，非实测”，没有物理净空认证。

历史正式tcp_target始终null，planning_allowed/executable=false。只有固定合成真值验证器可向隔离MoveIt发送PLAN_ONLY请求；本入口没有SDK、执行Action或控制话题。真实TCP、曝光映射、SDK/模型参考、工作场景、控制缺陷与事故锁仍阻塞实机。

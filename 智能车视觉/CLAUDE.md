# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

第21届 NXP 智能车竞赛 AI 视觉组项目。比赛模式：**虚实结合、硬件在环（Hardware-in-Loop）**。核心任务：用 **OpenART 摄像头**拍摄**车载手机屏幕**上的游戏画面（裁判系统 WiFi 投屏），通过图像识别+决策算法输出运动指令给 MCU，控制车模在 3.2m×2.4m 场地完成推箱子任务（当前拟定）。

**关键理解：摄像头拍屏幕，不是拍场地。** 屏幕上同时显示俯视图（全局）和第一人称视角（各 720×540）。车模在场地上的位置由裁判系统全局摄像头 + 车顶标识牌（含 AprilTag）定位。

## 硬件规范

- **车模**：B/C/F/G/M/H/H-mini/Y/Z 系列，投影 ≤ 35cm×35cm
- **主控**：限定 NXP MCU（推荐 i.MX RT 系列）
- **视觉模块**：OpenART / OpenART Plus / MCXVision / MV5-RT / OMV-RT
- **标识牌**：约 30cm×15cm，高度固定 15cm（磁吸），含红色矩形 + AprilTag + 灰绿色矩形 RGB(90,179,89)
- **场地 AprilTag**：编号 15-18（四角），高度 15cm
- **全局摄像头**：快门彩色，架设高度 ≥1.8m

## 技术栈

- **开发语言**: Python (PC 原型) / MicroPython (OpenART 部署)
- **视觉库**: OpenCV (PC 原型) / OpenMV sensor 模块 (OpenART)
- **通信**: UART 115200 8N1，与 MCU 通过自定 `$` 帧协议通信
- **目标帧率**: ≥20fps（识别+决策+发送需在 50ms 内完成）

## 通信协议

### 下行（发给 MCU）
| 格式 | 示例 | 说明 |
|------|------|------|
| `$V,vx,vy,wz*XX\n` | `$V,0.5,0,0.1*3A\n` | 速度指令，vx∈[-2,2] m/s, vy∈[-2,2] m/s, wz∈[-6,6] rad/s |
| `$S*XX\n` | `$S*53\n` | 紧急停车（不经减速） |
| `$R,x,y,yaw*XX\n` | `$R,1.5,2.0,1.57*0B\n` | 校正里程计位姿 |

### 上行（MCU 发出）
| 格式 | 频率 | 说明 |
|------|------|------|
| `$P,x,y,yaw,vx,vy,wz\r\n` | 50Hz | MCU 里程计位姿 |
| `$E,stall\r\n` | 事件 | 堵转/碰撞报警 |

### 校验
`*` 后两位十六进制 = `$` 到 `*` 之间所有字节 XOR。不带 `*` 也能解析（向后兼容）。

MCU 内部有加速度限制（0.8 m/s², 2 rad/s²），速度不会突变，可放心发大值。

## 两大游戏任务

### 推箱子 (Sokoban) — 当前比赛拟定任务
三种难度：传统推箱子 → 按图片分类推箱子 → 分类推箱子升级版。比赛支持闯关模式（单人计时）和对抗模式（双车同场）。
- 识别元素：车(红/蓝色块)、可移动箱子(橙黄 30×30px，可能贴卡通图案)、目的地(紫色 30×30px 带数字 0-9)、围墙(黑灰 20px)、炸弹
- 策略：绕到箱子背面 → 对准边界/目的地方向 → 直线推出
- 炸弹可炸内部围墙（外围围墙不可炸）
- 罚时：每未完成箱子 +30 秒，需完成半数以上
- 可参考练习系统（龙邱版本）：[CSDN 155713745](https://zhuoqing.blog.csdn.net/article/details/155713745)

### 飞机大战 (Plane Wars) — 练习/备赛系统
单人：击落 15 架敌机，3 条命，每掉 1 血 +10 秒罚时。双人：对抗模式，抢道具。
- 识别：敌机(1-3架)、敌方子弹、我方飞机、道具
- 策略：躲子弹优先 → 对准敌机 → 保持射击线
- 车模物理移动 = 飞机飞行轨迹，车模朝向 = 射击方向

## 开发路线（4-5周）

1. **PC + OpenCV 原型**（第1-2周）：USB 摄像头对屏幕，Python+OpenCV 识别游戏元素，串口发指令验证
2. **决策算法**（第2-3周）：推箱子路径规划（含炸弹策略）+ 飞机大战状态机，PC 模拟测试
3. **移植 OpenART**（第3-4周）：MicroPython 部署，算力优化
4. **联调优化**（第4-5周）：实车调试，帧率优化，参数调整

## 现有文件

- `main.py`：第17届国一队伍 OpenART 参考代码，含摄像头初始化、TF 模型分类、UART 通信、边缘检测
- `视觉.html`：完整技术方案文档（含通信协议细节、识别算法建议、参考代码）

## 识别方案

- **方案A（推荐起步）**：传统视觉 — HSV 颜色阈值分割 + 轮廓检测 + 模板匹配，快且 OpenART 能跑
- **方案B（进阶）**：轻量 CNN（YOLOv5-nano / MobileNet-SSD），量化 INT8 部署，鲁棒性好但需训练数据

## 硬件连接

- OpenART UART TX → MCU PA3 (USART2 RX)
- OpenART UART RX → MCU PA2 (USART2 TX)
- 波特率 115200, 8N1

## 关键约束

- WiFi 投屏有 50-100ms 延迟，决策需预判
- MCU 有 500ms 超时停车 + 堵转检测，代码崩溃车会自动停
- 用 HSV 而非 RGB 做阈值，应对现场光照变化
- 摄像头需避免屏幕反光，可加遮光罩
- 屏幕像素坐标 → 场地物理坐标 → 运动指令，需标定
- 标识牌已从黄色改为灰绿色 RGB(90,179,89)，避免与场地黄边框冲突

## 参考资料

| 资料 | 链接 |
|------|------|
| 比赛细则原文（卓晴） | [CSDN 154697884](https://blog.csdn.net/zhuoqingjoking97298/article/details/154697884) |
| 规则修订 | [CSDN 157844047](https://zhuoqing.blog.csdn.net/article/details/157844047) |
| Q&A 汇总 | [CSDN 南湖脑](https://nanhubrain.csdn.net/6a01534454b52172bc731a72.html) |
| 练习系统（龙邱） | [CSDN 155713745](https://zhuoqing.blog.csdn.net/article/details/155713745) |
| 组委会官网 | [smartcarrace.com](http://www.smartcarrace.com/) |

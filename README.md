# 黑洞构筑体 · Black Hole Construct 3D

基于六视图工程尺寸参考构建的 **黑洞构筑体（Black Hole Construct）** 高精度 3D 雕塑模型与交互式 WebGL 预览项目。

- **在线交互预览（GitHub Pages）**：https://xuzhiming.github.io/black-hole-construct-3d/

---

## 核心尺寸与建模规格

| 参数 | 图纸规范 | 模型实测 |
| :--- | :--- | :--- |
| **总包围盒 (`X × Y × Z`)** | `100 × 48 × 120 m` | `100.0 × 47.9 × 120.0 m` |
| **坐标系与原点** | 原点位于包围盒底面中心 (`Z = 0`)，正面朝 `-Y`，`Z` 向上，`1 BU = 1 m` | 完全对齐 |
| **洞口与视界黑核** | 洞心标高 `+58 m`，外径 `Ø40 m`，内黑核 `Ø26 m`，凹入深度 `18 m` | 凹面紫红能量裂隙 + 无反射纯黑 `EventHorizon` 内核 |
| **表面与卷须体系** | 多孔火山珊瑚暗黑曜石/氧化古铜外壳 + 背面海螺漩涡 + 113 根巴洛克螺旋卷须（根部 `Ø8–12 m` 至末端 `Ø0.5–1 m`） | 高模雕刻（~74 万面）→ 低模平滑拓扑高烘低 4K PBR |
| **WebGL 生产压缩资产** | `< 20 万面` + Draco 压缩 | **`145,540` 三角面** · **`14.17 MB`**（Draco L10 几何仅 `0.91 MB` + 双套 4K WebP PBR） |

---

## 网页预览功能

- **六视图一键切换**（底部栏或快捷键 `1`–`6`）：`正立面 FRONT`、`背立面 REAR`、`左立面 LEFT`、`右立面 RIGHT`、`俯视图 TOP`（自动切正交投影）与 `正交轴测`。
- **尺寸标注与图纸对比**：支持 3D 包围盒 / `Ø40m` / `Ø26m` / `18m` 凹深标注开关，以及原始六视图参考画中画对比。
- **实时材质与灯光控制**：支持 `PBR 材质` / `线框叠加` / `素模白模` / `纯线框` 模式切换，`宇宙暗棚` / `图纸米白` 背景切换，以及 Bloom 紫光辉光、曝光和主光旋转实时调节。

---

## 目录结构

```text
├── docs/                                # GitHub Pages 零依赖静态预览站点（17 MB）
│   ├── index.html                       # Three.js 交互预览主页面
│   ├── assets/
│   │   ├── black_hole_construct_draco200k.glb  # 14.5万面 Draco L10 + 双套 4K WebP 模型 (14.2 MB)
│   │   └── reference_6view.jpg          # 原始六视图尺寸参考图
│   └── vendor/three/                    # 精简本地 Three.js ES 模块与 Draco WASM 解码器
├── black_hole_construct.blend           # Blender 5.2 高精度雕刻源工程 (14 MB)
├── build_model.py                       # 参数化主壳 + 113 根无缝解析 UV 卷须建模脚本
├── export_draco_200k.py                 # 高模向低模 4K PBR 烘焙 + Draco L10 导出脚本
├── export_glb.py                        # 74万面双套无损 4K 母版导出脚本
└── start_preview.sh                     # 本地快速启动预览脚本
```

## 本地运行

```bash
python3 -m http.server 8765 --directory docs --bind 127.0.0.1
# 访问 http://127.0.0.1:8765/
```

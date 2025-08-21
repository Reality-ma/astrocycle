
# AstroCycle-MVP (v2)

将离散**深度–数值**数据转换到**时间域**并初筛**天文轨道周期旋回**（进动/倾角/短&长偏心率）。
此版本修复了导入路径问题：`app.py` 会自动把 `src/` 加入 `sys.path`，无需安装包即可在 **Streamlit Community Cloud** 或本地直接运行。

## 在线部署（Streamlit）
1. Fork 本仓库到你的 GitHub。
2. 在 https://share.streamlit.io 新建应用：`Branch: main`、`Main file path: app.py`。
3. 部署完成即可使用。

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 数据格式
- 主数据 CSV：`depth,value[,sigma]`
- 锚点 CSV（可选）：`depth,age_kyr[,age_sigma_kyr]`
示例在 `data/`。

## 功能
- 年龄–深度：PCHIP/线性；支持锚点年龄不确定度蒙特卡罗
- 非均匀 Lomb–Scargle 频谱 + 近似 AR(1) 红噪声95%阈
- 目标频带（19–23/41/100/405 kyr）带通 + Hilbert 包络
- 图形交互与下载（浏览器完成）

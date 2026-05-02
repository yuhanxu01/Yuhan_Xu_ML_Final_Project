# 快速使用指南

## 📋 文件夹内容
```
final_github_release/
├── README.md          # 项目说明
├── main.py            # 运行实验
├── setup.sh / setup.bat  # 自动下载数据
├── requirements.txt   # 依赖包
├── LICENSE           # MIT许可证
├── .gitignore        # Git忽略规则
└── src/              # 源代码(4个模块)
```

## 🚀 3步使用

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 自动下载数据集
```bash
# Windows用户
setup.bat

# Linux/Mac用户  
bash setup.sh
```

### 3. 运行实验
```bash
python main.py
```

## 📊 预期结果
- 运行时间: 15-30分钟
- 输出: results/comparison_*.png
- 显示: 3个策略的AA和BWT

## 🎯 实验数据

| 策略 | AA | BWT | 说明 |
|------|-----|-----|------|
| Naive | 0.724 | -0.189 | 基线，严重遗忘 |
| Frozen | 0.783 | -0.112 | 中等改善 |
| Replay | 0.847 | -0.043 | 最佳性能 |

## 💡 关键发现
- Experience Replay比基线提升12.3%
- 遗忘减少77.2%
- 最佳学习-遗忘平衡

## 📁 数据集自动下载
首次运行时自动下载UCI HAR数据集(30MB)，无需手动操作。

## ⚠️ 常见问题

**Q: 编译错误？**
A: 确保Python 3.8+，重新安装依赖

**Q: 数据下载失败？**
A: 检查网络，或手动下载放到data/目录

**Q: 运行时间太长？**
A: 可在main.py中减少epochs数量

## 📧 联系
Yuhan Xu  
Fordham University  
yxu32@fordham.edu

---

**🚀 直接拖拽此文件夹到GitHub即可发布！**

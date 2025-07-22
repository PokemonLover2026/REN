import matplotlib.pyplot as plt

# 示例数据
list1 = [1, 5, 3, 8, 7]
list2 = [2, 3, 6, 5, 9]

# 获取列表长度
n = len(list1)

# 创建idx列表
idx = list(range(n))

# 绘制点云图
plt.figure()

# 分别绘制list1和list2中的点
plt.scatter(idx, list1, color='blue', label='List 1')
plt.scatter(idx, list2, color='red', label='List 2')

# 添加图例
plt.legend()

# 设置标题和坐标轴标签
plt.title('点云图')
plt.xlabel('索引 (idx)')
plt.ylabel('值 (value)')

# 保存图表
plt.savefig('pointcloudplot.png')
import cvxpy as cp
import numpy as np
from sympy import symbols, expand

# 定义变量 X
n = 2
X = cp.Variable((n, n), symmetric=True)

# 定义目标矩阵 C
C = np.array([[1, 2], [3, 4]])

# 定义目标函数
objective = cp.Maximize(cp.trace(C.T @ X))

# 定义约束
constraints = [
    X >> 0,  # X 应该是半正定的
    X << np.eye(n)  # X 应该小于等于单位矩阵
]

# 构建并求解问题
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)

# 输出结果
print("最优值:", prob.value)
print("最优解 X:\n", X.value)



# # 定义未知数x
# x = symbols('x')
#
# # 定义两个方程的系数，其中c和f是未知数x的函数
# a, b, c, d, e, f = symbols('a b c d e f')
# a = b = c = d = e = 1
# coefficients1 = [a*x**2, b*x, c]
# coefficients2 = [d*x**2, e*x, f]
#
# # 构建方程
# equation1 = sum(coefficients1)
# equation2 = sum(coefficients2)
#
# # 计算乘积并展开
# product = expand(equation1 * equation2)
#
# # 输出结果
# print(product)
import numpy as np
import matplotlib.pyplot as plt

# x 구간 정의
x = np.linspace(0, 3, 300)

# C 값을 임의로 1로 설정 (문제에서 실제로는 정규화 필요)
C = 9/2

# PDF 정의
pdf = C * (x + 0.5 * x**2 - 1)

# 음수값은 0으로 처리 (0 <= x <= 3 범위만 사용)
pdf = np.where((x >= 0) & (x <= 3), pdf, 0)

plt.figure(figsize=(8, 4))
plt.plot(x, pdf, label='PDF: $C(x + 0.5x^2 - 1)$')
plt.xlabel('$x$')
plt.ylabel('$f_X(x)$')
plt.title('Probability Density Function')
plt.legend()
plt.grid(True)
plt.show()

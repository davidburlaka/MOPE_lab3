from numpy import *
from math import *
import numpy as np

m = int(input("Введіть m: "))

rows = 4
x1_min, x1_max = 15, 45
x2_min, x2_max = 30, 80
x3_min, x3_max = 15, 45
x_avarage_max = (x1_max + x2_max + x3_max) / 3
x_avarage_min = (x1_min + x2_min + x3_min) / 3
y_max = 200 + x_avarage_max
y_min = 200 + x_avarage_min

# матриця кодованих значень х
matrix_x_cod = [[+1, -1, -1, -1],
                [+1, -1, +1, +1],
                [+1, +1, -1, +1],
                [+1, +1, +1, -1]]

# матриця значень х
matrix_x = np.matrix(
    [[x1_min, x2_min, x3_min],
     [x1_min, x2_max, x3_max],
     [x1_max, x2_min, x3_max],
     [x1_max, x2_max, x3_min]])

# матриця рандомних значень у
random_matrix_y = random.randint(y_min, y_max, size=(rows, m))


# сума середніх значень відгуку функції за рядками
def sum_rows(random_matrix_y):
    y = np.sum(random_matrix_y, axis=1) / rows
    return y


y1 = sum_rows(random_matrix_y)


# Нормовані коефіціенти рівняння регресії
def sum_mx(a, b, c, d):
    mx = (a + b + c + d) / rows
    return mx


mx1 = sum_mx(x1_min, x1_min, x1_max, x1_max)
mx2 = sum_mx(x2_min, x2_max, x2_min, x2_max)
mx3 = sum_mx(x3_min, x3_max, x3_max, x3_min)


# Нормовані коефіціенти рівняння регресії
def sum_my(a, b, c, d):
    my = (a + b + c + d) / rows
    return my


my = sum_my(y1[0], y1[1], y1[2], y1[3])


# Нормовані коефіціенти рівняння регресії
def find_a(a, b, c, d):
    az = (a * y1[0] + b * y1[1] + c * y1[2] + d * y1[3]) / rows
    return az


a1 = find_a(x1_min, x1_min, x1_max, x1_max)
a2 = find_a(x2_min, x2_max, x2_min, x2_max)
a3 = find_a(x3_min, x3_max, x3_max, x3_min)


# Нормовані коефіціенти рівняння регресії
def find_aa(a, b, c, d):
    aa = (a ** 2 + b ** 2 + c ** 2 + d ** 2) / rows
    return aa


a11 = find_aa(x1_min, x1_min, x1_max, x1_max)
a22 = find_aa(x2_min, x2_max, x2_min, x2_max)
a33 = find_aa(x3_min, x3_max, x3_max, x3_min)

# Нормовані коефіціенти рівняння регресії
a12 = a21 = (x1_min * x2_min + x1_min * x2_max + x1_max * x2_min + x1_max * x2_max) / rows
a13 = a31 = (x1_min * x3_min + x1_min * x3_max + x1_max * x3_max + x1_max * x3_min) / rows
a23 = a32 = (x2_min * x3_min + x2_max * x3_max + x2_min * x3_max + x2_max * x3_min) / rows

# Матриця для визначення коефіціентів регресії
A = [[my, mx1, mx2, mx3], [a1, a11, a12, a13], [a2, a12, a22, a32], [a3, a13, a23, a33]]
B = [[1, my, mx2, mx3], [mx1, a1, a12, a13], [mx2, a2, a22, a32], [mx3, a3, a23, a33]]
C = [[1, mx1, my, mx3], [mx1, a11, a1, a13], [mx2, a12, a2, a32], [mx3, a13, a3, a33]]
D = [[1, mx1, mx2, my], [mx1, a11, a12, a1], [mx2, a12, a22, a2], [mx3, a13, a23, a3]]
E = [[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a32], [mx3, a13, a23, a33]]
X = []


# Коефіціенти регресії
def coef_regr(a, b):
    b = linalg.det(a) / linalg.det(b)
    return b


b0 = coef_regr(A, E)
b1 = coef_regr(B, E)
b2 = coef_regr(C, E)
b3 = coef_regr(D, E)
X.append(round(b0, 2))
X.append(round(b1, 2))
X.append(round(b2, 2))
X.append(round(b3, 2))


# Нормоване рівняння регресії
def find_y_norm(a, b, c):
    y_norm = X[0] + X[1] * a + X[2] * b + X[3] * c
    return y_norm


y_norm1 = find_y_norm(x1_min, x2_min, x3_min)
y_norm2 = find_y_norm(x1_min, x2_max, x3_max)
y_norm3 = find_y_norm(x1_max, x2_min, x3_max)
y_norm4 = find_y_norm(x1_max, x2_max, x3_min)


# Перевірка однорідності дисперсії за критерієм Кохрена
# Пошук дисперсій по рядкам
def find_s(a, b, c):
    s = ((a - y1[0]) ** 2 + (b - y1[0]) ** 2 + (c - y1[0]) ** 2) / 3
    return s


s1 = find_s(random_matrix_y[0][0], random_matrix_y[0][1], random_matrix_y[0][2])
s2 = find_s(random_matrix_y[1][0], random_matrix_y[1][1], random_matrix_y[1][2])
s3 = find_s(random_matrix_y[2][0], random_matrix_y[2][1], random_matrix_y[2][2])
s4 = find_s(random_matrix_y[3][0], random_matrix_y[3][1], random_matrix_y[3][2])

Gp = max(s1, s2, s3, s4) / (s1 + s2 + s3 + s4)
f1 = m - 1
f2 = rows
Gt = 0.7679
# Перевірка умови за критерієм Кохрена
uniform = Gp <= Gt

# Оцінимо значимість коефіціентів за критерієм Стьюдента
Sb = (s1 + s2 + s3 + s4) / rows
Sbetakvadr = Sb / (rows * m)
Sbeta = sqrt(Sb / (rows * m))


# Визначимо оцінки коефіціентів
def find_beta(a, b, c, d):
    beta = (y1[0] * a + y1[1] * b + y1[2] * c + y1[3] * d) / rows
    return beta


beta0 = find_beta(matrix_x_cod[0][0], matrix_x_cod[1][0], matrix_x_cod[2][0], matrix_x_cod[3][0])
beta1 = find_beta(matrix_x_cod[0][1], matrix_x_cod[1][1], matrix_x_cod[2][1], matrix_x_cod[3][1])
beta2 = find_beta(matrix_x_cod[0][2], matrix_x_cod[1][2], matrix_x_cod[2][2], matrix_x_cod[3][2])
beta3 = find_beta(matrix_x_cod[0][3], matrix_x_cod[1][3], matrix_x_cod[2][3], matrix_x_cod[3][3])


# Пошук коефіціента t
def find_t(a, b):
    t = a / b
    return t


t0 = find_t(beta0, Sbeta)
t1 = find_t(beta1, Sbeta)
t2 = find_t(beta2, Sbeta)
t3 = find_t(beta3, Sbeta)
t_list = [fabs(t0), fabs(t1), fabs(t2), fabs(t3)]
b_list = [b0, b1, b2, b3]
t_tabl = 2.306

# Перевірка умови за критерієм Стьюдента
for i in range(4):
    if t_list[i] < t_tabl:
        t_list[i] = 0

for j in range(4):
    if t_list[j] == 0:
        b_list[j] = 0


# Запишемо рівняння з урахуванням критерію Стьюдента
def find_yj(a, b, c):
    yj = b_list[0] + b_list[1] * a + b_list[2] * b + b_list[3] * c
    return yj

yj1 = find_yj(x1_min, x2_min, x3_min)
yj2 = find_yj(x1_min, x2_max, x3_max)
yj3 = find_yj(x1_max, x2_min, x3_max)
yj4 = find_yj(x1_max, x2_max, x3_min)

# Критерій Фішера
d = 1  # кількість значимих коефіціентів
f4 = rows - d
f3 = f1 * f2
Sad = m * (((yj1 - y1[0]) ** 2 + (yj2 - y1[1]) ** 2 + (yj3 - y1[2]) ** 2 + (yj4 - y1[3]) ** 2)) / f4
Fp = Sad / Sbetakvadr
Ft = 5.3

print("\n")
print("Рівняння регресії: ŷ = b0 + b1*x1 + b2*x2+ b3*x3 ")
print("Ymax: ", round(x_avarage_max, 2))
print("Ymin:", round(x_avarage_min, 2))
print("\n")
print("Матриця кодованих значень Х: \n", matrix_x_cod)
print("Матриця для значень Х: \n", matrix_x)
print("\n")
print("Матриця для значень Y: \n", random_matrix_y)
print("\n")
print("y1: ", y1[0], "\ty2: ", y1[1], "\ty3: ", y1[2], "\ty4: ", y1[3])
print("mx1:", mx1, "\tmx2:", mx2, "\tmx3:", mx3)
print("my: ", my)
print("\n")
print("b0: ", X[0], "\tb1: ", X[1], "\tb2: ", X[2], "\tb3: ", X[3])
print("\n")
print("Нормоване рівняння регресії y =", X[0], "+", X[1], "* x1 +", X[2], "* x2")
print(X[0], "+", X[1] * x1_min, "+", X[2] * x2_min, "+", X[3] * x3_min, "=", round(y_norm1, 2))
print(X[0], "+", X[1] * x1_min, "+", X[2] * x2_max, "+", X[3] * x3_max, "=", round(y_norm2, 2))
print(X[0], "+", X[1] * x1_max, "+", X[2] * x2_min, "+", X[3] * x3_max, "=", round(y_norm3, 2))
print(X[0], "+", X[1] * x1_max, "+", X[2] * x2_max, "+", X[3] * x3_min, "=", round(y_norm4, 2))
print("\n")
print("Перевірка за Кохреном")
print("S²{y1}: ", round(s1, 2))
print("S²{y2}: ", round(s2, 2))
print("S²{y3}: ", round(s3, 2))
print("S²{y4}: ", round(s4, 2))
print("Gp: ", Gp)
print("Диперсія однорідна:", uniform)
print("\n")
print("Перевірка за Стьюдентом")
print("Sb²: ", round(Sb, 2))
print("S²{β}: ", round(Sbetakvadr, 2))
print("S{β}: ", round(Sbeta, 2))
print("β1: ", beta0, "\tβ2: ", beta1, "\tβ3: ", beta2, "\tβ4: ", beta3)
print("t0: ", round(t0, 2))
print("t1: ", round(t1, 2))
print("t2: ", round(t2, 2))
print("t3: ", round(t3, 2))
print("ŷ1 =", round(yj1, 2))
print("ŷ2 =", round(yj2, 2))
print("ŷ3 =", round(yj3, 2))
print("ŷ4 =", round(yj4, 2))
print("\n")
print("Перевірка за Фішером")
print("Sad²: ", round(Sad, 2))
print("Fp: ", round(Fp, 2))
if Fp > Ft:
    print("Pівняння регресії неадекватно оригіналу при рівні значимості 0.05")
else:
    print("Pівняння регресії адекватно оригіналу при рівні значимості 0.05")

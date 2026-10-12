import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def tabulate(f, x0, xn, n, filename):
    h = (xn - x0) / n
    with open(filename, 'w') as file:
        for i in range(n + 1):
            x = x0 + i * h
            file.write("%.10f %.10f\n" % (x, f(x)))


def read_table(filename):
    x = []
    y = []
    with open(filename, 'r') as file:
        for line in file:
            a, b = line.split()
            x.append(float(a))
            y.append(float(b))
    return x, y


def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return x, y

def omega(k, t, x):
    w = 1.0
    for i in range(k + 1):
        w *= (t - x[i])
    return w


def div_diff(x, f, k):
    s = 0.0
    for i in range(k + 1):
        p = 1.0
        for j in range(k + 1):
            if j != i:
                p *= (x[i] - x[j])
        s += f[i] / p
    return s


def div_diff_table(x, f):
    n = len(x)
    table = [list(f)]
    for k in range(1, n):
        prev = table[k - 1]
        row = []
        for i in range(n - k):
            row.append((prev[i + 1] - prev[i]) / (x[i + k] - x[i]))
        table.append(row)
    return table


def print_table(x, table):
    print("x_i".ljust(14) + "".join(("f[%d]" % k).ljust(16) for k in range(len(table))))
    for i in range(len(x)):
        line = ("%.4f" % x[i]).ljust(14)
        for k in range(len(table)):
            if i < len(table[k]):
                line += ("%.6g" % table[k][i]).ljust(16)
        print(line)


def newton(t, x, f):
    n = len(x) - 1
    res = f[0]
    for k in range(1, n + 1):
        res += omega(k - 1, t, x) * div_diff(x, f, k)
    return res


def error(func, t, x, f):
    return abs(func(t) - newton(t, x, f))

def fact_poly(t, k):
    res = 1.0
    for i in range(k):
        res *= (t - i)
    return res


def finite_diffs_at_zero(f):
    d = list(f)
    res = [d[0]]
    for k in range(1, len(f)):
        d = [d[i + 1] - d[i] for i in range(len(d) - 1)]
        res.append(d[0])
    return res


def factorial_interp(xv, x, f):
    h = x[1] - x[0]
    t = (xv - x[0]) / h
    d = finite_diffs_at_zero(f)
    res = 0.0
    for k in range(len(f)):
        res += d[k] / math.factorial(k) * fact_poly(t, k)
    return res

def grid(a, b, n):
    m = 20 * n
    return [a + (b - a) * i / m for i in range(m + 1)]


def choose_nodes(x, f, xv, n):
    idx = sorted(range(len(x)), key=lambda i: abs(x[i] - xv))[:n + 1]
    idx.sort()
    return [x[i] for i in idx], [f[i] for i in idx]

def test_part():
    func = math.sin
    a, b = 0.0, 3.0

    for n in (5, 10, 20):
        tabulate(func, a, b, n, "sin_%d.txt" % n)
        x, f = read_table("sin_%d.txt" % n)

        tt = grid(a, b, n)
        N = [newton(t, x, f) for t in tt]
        eps = [error(func, t, x, f) for t in tt]
        w = [omega(n, t, x) for t in tt]

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(tt, [func(t) for t in tt], label='f(x)')
        plt.plot(tt, N, '--', label='N_n(x)')
        plt.plot(x, f, 'ro')
        plt.title("n = %d" % n)
        plt.legend()
        plt.grid()
        plt.subplot(2, 2, 2)
        plt.plot(tt, eps)
        plt.title("Похибка eps(x)")
        plt.grid()
        plt.subplot(2, 2, 3)
        plt.plot(tt, w)
        plt.title("w_n(x)")
        plt.grid()
        plt.tight_layout()
        plt.savefig("sin_n%d.png" % n)
        plt.show()
        print("n = %d, max eps = %.3e" % (n, max(eps)))

    print("\nФіксований інтервал [0, 3], різне n:")
    for n in (5, 10, 20):
        x, f = read_table("sin_%d.txt" % n)
        tt = grid(a, b, n)
        m = max(error(func, t, x, f) for t in tt)
        print("n = %2d, h = %.4f, max eps = %.3e" % (n, (b - a) / n, m))

    print("\nФіксований крок h = 0.3, змінний інтервал:")
    h = 0.3
    for n in (5, 10, 20):
        bb = a + h * n
        tabulate(func, a, bb, n, "sin_h_%d.txt" % n)
        x, f = read_table("sin_h_%d.txt" % n)
        tt = grid(a, bb, n)
        m = max(error(func, t, x, f) for t in tt)
        print("n = %2d, [%.1f, %.1f], max eps = %.3e" % (n, a, bb, m))

def variant_part():
    x, y = read_data("data.csv")      # x - кількість об'єктів, y - FPS
    xv = 1000.0

    table = div_diff_table(x, y)
    print("\nТаблиця розділених різниць:")
    print_table(x, table)

    p_newton = newton(xv, x, y)
    print("\nFPS(1000) за Ньютоном (5 вузлів) = %.4f" % p_newton)

    x0, xn = x[0], x[-1]
    for n in (5, 10, 20):
        h = (xn - x0) / n
        xs = [x0 + i * h for i in range(n + 1)]
        ys = [newton(t, x, y) for t in xs]
        p_fact = factorial_interp(xv, xs, ys)
        print("n = %2d, h = %7.2f, FPS(1000) факторіальний = %.4f" % (n, h, p_fact))

    a, b = 800.0, 1600.0
    while b - a > 1e-6:
        c = (a + b) / 2
        if (newton(a, x, y) - 60) * (newton(c, x, y) - 60) <= 0:
            b = c
        else:
            a = c
    n60 = (a + b) / 2
    print("\nFPS = 60 при n = %.2f об'єктів" % n60)
    print("Для FPS >= 60 потрібно n <= %d" % math.floor(n60))

    tt = list(np.linspace(x0, xn, 400))
    plt.figure(figsize=(8, 5))
    plt.plot(tt, [newton(t, x, y) for t in tt], label='P(n)')
    plt.plot(x, y, 'ro', label='експеримент')
    plt.plot([xv], [p_newton], 'g*', markersize=12, label='FPS(1000)')
    plt.axhline(60, color='gray', linestyle='--')
    plt.xlabel('Objects')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid()
    plt.savefig("variant5.png")
    plt.show()

    ref = lambda t: float(np.interp(t, x, y))
    print("\nПрогноз FPS(1000) при різній кількості вузлів (еталон %.4f):" % ref(xv))
    for n in (1, 2, 3, 4):
        xs, ys = choose_nodes(x, y, xv, n)
        p = newton(xv, xs, ys)
        print("n = %d, FPS(1000) = %.4f, eps = %.4f" % (n, p, abs(p - ref(xv))))

    plt.figure(figsize=(8, 5))
    for n in (5, 10, 20):
        h = (xn - x0) / n
        xs = [x0 + i * h for i in range(n + 1)]
        ys = [ref(t) for t in xs]
        eps = [abs(ref(t) - newton(t, xs, ys)) for t in tt]
        print("n = %2d, max eps = %.4f" % (n, max(eps)))
        plt.plot(tt, eps, label='n = %d' % n)
    plt.xlabel('Objects')
    plt.ylabel('eps')
    plt.legend()
    plt.grid()
    plt.title("Похибка інтерполяції")
    plt.savefig("errors.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    for n in (5, 10, 20):
        h = (xn - x0) / n
        xs = [x0 + i * h for i in range(n + 1)]
        ys = [ref(t) for t in xs]
        plt.plot(tt, [newton(t, xs, ys) for t in tt], label='n = %d' % n)
    plt.plot(x, y, 'ko')
    plt.ylim(0, 160)
    plt.legend()
    plt.grid()
    plt.title("Ефект Рунге")
    plt.savefig("runge.png")
    plt.show()


if __name__ == "__main__":
    test_part()
    variant_part()
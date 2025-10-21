import math

def find_third_side():

    a = float(input("Введите длину первой стороны: "))
    b = float(input("Введите длину второй стороны: "))
    angle_degrees = float(input("Введите угол между сторонами в градусах: "))

    angle_radians = math.radians(angle_degrees)

    c_squared = a ** 2 + b ** 2 - 2 * a * b * math.cos(angle_radians)
    c = math.sqrt(c_squared)
    print(f"Длина третьей стороны: {c:.2f}")

find_third_side()
# Запрашиваем у пользователя число N
# Обрабатываем некорректный ввод
while True:
    try:
        n_input = input("Введите число N (не меньше 2): ")
        n = int(n_input) # Пробуем преобразовать ввод в целое число
        if n < 2:
            print("Число должно быть 2 или больше. Попробуйте снова.")
        else:
            break # Если ввод корректный, выходим из цикла
    except ValueError: # Если преобразование в int не удалось
        print("Пожалуйста, введите целое число.")

# Создаем список, где будем отмечать, является ли число простым
# True означает, что число может быть простым
# Сначала предполагаем, что все числа от 0 до N - простые
is_prime = [True] * (n + 1)

# 0 и 1 не являются простыми числами, помечаем их как False
is_prime[0] = False
is_prime[1] = False

# Решето Эратосфена
# Начинаем с 2, первого простого числа
current_number = 2
while current_number * current_number <= n: # Проверяем только до корня из N
    # Если current_number помечен как простое
    if is_prime[current_number]:
        # Помечаем все числа, кратные current_number (начиная с current_number*current_number), как не простые
        multiple = current_number * current_number
        while multiple <= n:
            is_prime[multiple] = False
            multiple += current_number # Переходим к следующему кратному числу
    # Переходим к следующему числу
    current_number += 1

# Выводим все числа, которые остались отмеченными как простые
print(f"Простые числа от 2 до {n}:")
for number in range(2, n + 1):
    if is_prime[number]:
        print(number)

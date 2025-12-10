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

is_prime = [True] * (n + 1)
is_prime[0] = False
is_prime[1] = False

current_number = 2
while current_number * current_number <= n:
    if is_prime[current_number]:
        multiple = current_number * current_number
        while multiple <= n:
            is_prime[multiple] = False
            multiple += current_number
    current_number += 1

print(f"Простые числа от 2 до {n}:")
for number in range(2, n + 1):
    if is_prime[number]:
        print(number)

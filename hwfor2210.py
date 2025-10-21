# Просим пользователя ввести температуру
s = input("Введите температуру (например, 40C или 100F): ")

# Убираем пробелы по краям
s = s.strip()

# Проверим, что строка не пустая
if len(s) == 0:
    print("Некорректный ввод")
else:
    # Последний символ — это единица измерения?
    last_char = s[-1]
    if last_char != 'C' and last_char != 'F':
        print("Некорректный ввод")
    else:
        # Остальная часть строки — это число?
        number_str = s[0:-1]

        # Проверим, что это число (и содержит хотя бы один символ)
        if len(number_str) == 0:
            print("Некорректный ввод")
        else:
            # Попробуем преобразовать в число
            is_number = True
            for c in number_str:
                if c != '-' and c != '.' and not c.isdigit():
                    is_number = False
                    break

            if not is_number:
                print("Некорректный ввод")
            else:
                t = float(number_str)

                # Теперь конвертируем
                if last_char == 'C':
                    # Цельсий в Фаренгейт
                    new_t = t * 9 / 5 + 32
                    print(f"{new_t:g}F")
                else:  # last_char == 'F'
                    # Фаренгейт в Цельсий
                    new_t = (t - 32) * 5 / 9
                    print(f"{new_t:g}C")
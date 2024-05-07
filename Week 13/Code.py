def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

num = int(input("Nhập số cần tính giai thừa: "))
print("Giai thừa của", num, "là", factorial(num))
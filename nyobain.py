inp = int(input('Masukkan Sampel => '))
table = []
for i in range(1, inp+1):
    temp = []
    for y in range(1, inp+1):
        temp.append(y*i)
        ends = " "
    table.append(temp)
print(table)
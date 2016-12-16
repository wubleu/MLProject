f = open("output_17500","r")
total = 0
lines = f.readlines()
for i,line in enumerate(lines):
    line = line.strip()
    if i % 50 == 0 and i > 0:
        print("("+str(i)+","+str(total)+")")
        total = 0
    total += int(line)
f.close()
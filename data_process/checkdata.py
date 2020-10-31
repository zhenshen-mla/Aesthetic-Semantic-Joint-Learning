import numpy as np

indexPath = '/home1/shenzhen/Downloads/ava/AVA_dataset/AVA.txt'
file = open(indexPath, 'r')
lines = file.readlines()
file.close()

list_index0 = []
list_index1 = []
list_index1_val = []
list_aes0 = []
list_aes1 = []
list_aes1_val = []
list_seg0 = []
list_seg1 = []
list_seg1_val = []

delta = 0  # this is the critical item for option
lll = [953980, 440774, 954113, 953958, 953619, 953349, 954175, 953897, 310261, 953841, 179118, 371434, 848725, 567829, 277832]
c1 = 0
c2 = 0
l = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for line in lines:
    temp = line.split(' ')
    if int(temp[12]) > 0 and int(temp[12]) <= 66 and int(temp[13])== 0:
        c1 += 1
        l[int(temp[12])-1] += 1



print(c1)
print(l)  # 45915
target = [0,4,13,14,16,17,18,19,20,21,37,39]
for i in l:
    if i>=1000:
        print(i, l.index(i))




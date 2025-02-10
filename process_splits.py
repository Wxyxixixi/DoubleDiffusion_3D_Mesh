import os
import shutil

# list_eval_partition.txt:
# 000001.jpg 0
# 000002.jpg 0
# 000003.jpg 0

# mapping.txt:
# 0          119613     119614.jpg
# 1          99094      099095.jpg
# 2          200121     200122.jpg
# 3          81059      081060.jpg

list_eval_partition_dict = dict()
list_eval_partition_dict['0'] = []
list_eval_partition_dict['1'] = []
list_eval_partition_dict['2'] = []

list_eval_partition_reversed = dict()

# with open('/ibex/ai/home/liz0l/projects/codes/ddpm_diffusionNet/splits/list_eval_partition.txt', 'r') as f:
with open('/splits/list_eval_partition.txt', 'r') as f:
    list_eval_partition_list = f.readlines()
    for line in list_eval_partition_list:
        line = line.strip().split(' ')
        list_eval_partition_dict[line[1]].append(line[0])
        list_eval_partition_reversed[line[0]] = line[1]
        


mapping_dict = dict()
# with open('/ibex/ai/home/liz0l/projects/codes/ddpm_diffusionNet/splits/mapping.txt', 'r') as f:
with open('splits/mapping.txt', 'r') as f:
    mapping_list = f.readlines()
    for line in mapping_list:
        line = line.strip().split(' ')
        mapping_dict[line[0]] = line[-1]
        
        
train_set = []
test_set = []
for i in range(30000):
    corresponding_image = mapping_dict[str(i)]
    corresponding_image_set = list_eval_partition_reversed[corresponding_image]
    if corresponding_image_set == '0':
        train_set.append("{}.jpg".format(str(i)))
    elif corresponding_image_set == '2':
        test_set.append("{}.jpg".format(str(i)))
    else:
        continue

print(len(train_set))
print(len(test_set))

# with open('/ibex/ai/home/liz0l/projects/codes/ddpm_diffusionNet/splits/train_set.txt', 'w') as f:
with open('splits/train_set.txt', 'w') as f:
    for item in train_set:
        f.write("%s\n" % item)

# with open('/ibex/ai/home/liz0l/projects/codes/ddpm_diffusionNet/splits/test_set.txt', 'w') as f:
with open('splits/test_set.txt', 'w') as f:
    for item in test_set:
        f.write("%s\n" % item)

for item in train_set:
    shutil.copy("/ibex/ai/home/liz0l/projects/codes/ddpm_diffusionNet/datasets/images/celeba_hq/{}".format(item), "/ibex/ai/home/liz0l/projects/codes/ddpm_diffusionNet/datasets/images/celeba_hq_train/{}".format(item))
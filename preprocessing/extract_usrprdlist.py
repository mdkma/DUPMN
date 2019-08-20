filename_train = '../data/ca_DATA/re_sampled_ca_comment_train.txt'
filename_test = '../data/ca_DATA/re_sampled_ca_comment_test.txt'

filename_usrlist_save = '../data/ca_DATA/usrlist.txt'
filename_prdlist_save = '../data/ca_DATA/prdlist.txt'

lines = list(map(lambda x: x.split('\t\t'), open(filename_train, "r", encoding='utf-8').readlines()))
print ('num of lines in train: ', len(lines))

usr_all_train = list(map(lambda line: line[0],lines))
prd_all_train = list(map(lambda line: line[1],lines))
print ('num of usr_all_train: ', len(usr_all_train))
print ('num of prd_all_train: ', len(prd_all_train))

lines = list(map(lambda x: x.split('\t\t'), open(filename_test, "r", encoding='utf-8').readlines()))
print ('num of lines in test: ', len(lines))

usr_all_test = list(map(lambda line: line[0],lines))
prd_all_test = list(map(lambda line: line[1],lines))
print ('num of usr_all_test: ', len(usr_all_test))
print ('num of prd_all_test: ', len(prd_all_test))

usr_uniq = list(set(usr_all_train + usr_all_test))
prd_uniq = list(set(prd_all_train + usr_all_test))

print ('num of uniq usr: ', len(usr_uniq))
print ('num of uniq prd: ', len(prd_uniq))

usr_text = '\n'.join(usr_uniq) + '\n'
prd_text = '\n'.join(prd_uniq) + '\n'

f = open(filename_usrlist_save,"w+")
f.write(usr_text)
f.close()

f = open(filename_prdlist_save,"w+")
f.write(prd_text)
f.close()
print ('-> saved')
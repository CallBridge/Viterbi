# import submission as submission
import submission as submission


State_File ='./dev_set/State_File'
Symbol_File='./dev_set/Symbol_File'
Query_File ='./dev_set/Query_File'
Query_Label ='./dev_set/Query_Label'


viterbi_result = submission.advanced_decoding(State_File, Symbol_File, Query_File)

result_list=[]
for line in viterbi_result:
    result_list.append(line[:-1])
l=[]
with open('./dev_set/Query_Label') as f:
    for line in f:
        l.append(line.split())
count=0
for i in range(len(l)):
    for j in range(len(l[i])):
        if result_list[i][j]!=int(l[i][j]):
            count+=1

print("Margin is:", 134-count)


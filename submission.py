# Import your files here...
import pandas as pd
import numpy as np


# Has two dictionarys
# one is state name -> state id
# one is the transition frequency (see its comment)
class State:
    def __init__(self, file_name):
        # State part
        self.states = {}  # Key is descriptive name, value is ID.
        self.state_num = 0
        self.transitions = {}  # Key: (f1, f2), value f3
        self.build_states(file_name)

    def build_states(self, file_name):
        with open(file_name) as f:
            lines = f.readlines()
            # Obtain state numbers, ie. how many kinds of state do we have
            self.state_num = int(lines[0])
            state_id = 0
            # Obtain states
            for i in range(1, self.state_num + 1):
                self.states[self.clip_name(lines[i])] = state_id
                state_id += 1
            # Obtain state transitions
            for i in range(self.state_num + 1, len(lines)):
                each_transi = lines[i].split(' ')
                f1 = each_transi[0]
                f2 = each_transi[1]
                frequency = self.clip_name(each_transi[2])
                self.transitions[(int(f1), int(f2))] = int(frequency)

    # Since there are space and new line, so should clip strings
    def clip_name(self, str):
        i = len(str) - 1
        end_index = 0
        while i >= 0:
            if str[i] != ' ' and str[i] != '\n':
                end_index = i
                break
            i -= 1
        return str[0:end_index + 1]

    # Obtain a state's descriptive name corresponding to id
    def get_state_id(self, name):
        return self.states.get(name)

    #Obtain a state's descriptive name corresponding to id
    def get_state_name(self, state_id):
        return self.states.get(state_id)


# One dict for name -> symbol id
# one dict for emisssion
class Symbol:
    def __init__(self, Symbol_File):
        self.symbols = {}  # K: descriptive name, V: ID
        self.sym_num = 0  # how many symbols are there
        self.emissions = {}  # (state, symbol):frequency
        self.build_symbol(Symbol_File)

    # Since there are space and new line, so should clip strings
    def clip_name(self, str):
        i = len(str) - 1
        end_index = 0
        while i >= 0:
            if str[i] != ' ' and str[i] != '\n':
                end_index = i
                break
            i -= 1
        return str[0:end_index + 1]

    def build_symbol(self, Symbol_File):
        with open(Symbol_File) as f:
            lines = f.readlines()
            self.sym_num = int(lines[0])
            symbol_id = 0
            for i in range(1, self.sym_num + 1):
                self.symbols[self.clip_name(lines[i])] = symbol_id
                symbol_id += 1
            for i in range(self.sym_num + 1, len(lines)):
                each_emit = lines[i].split(' ')
                f1 = each_emit[0]
                f2 = each_emit[1]
                frequency = self.clip_name(each_emit[2])
                self.emissions[(int(f1), int(f2))] = int(frequency)

    def get_symbol_id(self, name):
        return self.symbols.get(name)

class Query:
    def __init__(self, Query_File):
        self.all_query = []
        self.lex_all(Query_File)
        self.translated_Q = []

    def lex_all(self, Query_File):
        with open(Query_File, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                self.all_query.append(self.lex_single_query(l))

    def lex_single_query(self, str):
        tkn = []
        rc = 0  # right cursor
        lc = 0  # left cursor
        while rc < len(str):
            # Collect the puncutation
            if self.is_punctuation(str[rc]):
                tkn.append(str[lc:rc + 1])
                rc += 1
                lc += 1
            # skip spaces
            elif str[rc] == ' ':
                while str[rc] == ' ':
                    rc += 1
                    if rc >= len(str):
                        break
                lc = rc
            else:
                while self.is_punctuation(str[rc]) == False and str[rc] != ' ':
                    rc += 1
                    if rc >= len(str):
                        break
                tkn.append(str[lc:rc])
                lc = rc
        return tkn

    def is_punctuation(self, char):
        if char == '*' or char == '(' or char == ')' or char == '/' or char == '-' or char == '&' or char == ',':
            return True
        else:
            return False

    def is_part_of_string(self, char):
        return (char >= '0' and char <= '9') or (char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z')

    def token_to_sym_id(self, symbol_table):
        translated_single_Q = {}
        for i in self.all_query:
            for j in i:
                if j not in symbol_table.keys():
                    translated_single_Q[j] = 'UNK'
                else:
                    translated_single_Q[j] = symbol_table[j]
            self.translated_Q.append(translated_single_Q)
            translated_single_Q = {}

# All stufff is in label_to_state
# it's a list of dictionaries
# each dictionary is each line in he query_label file
# dic: state_id : state_name
class Query_Labels:
    def __init__(self, Query_Label):
        self.Label_to_state = []
        self.lex_all(Query_Label)

    def lex_all(self, Query_Label):
        with open(Query_Label, 'r') as f:
            lines = f.read().splitlines()
            for s in lines:
                L = s.split(' ')
                self.Label_to_state.append([int(e) for e in L])


# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File):  # do not change the heading of the function
    # init file parser
    state = State(State_File)
    symbol = Symbol(Symbol_File)
    queries = Query(Query_File)
    queries.token_to_sym_id(symbol.symbols)
    # smoothing data from parser

    N=len(state.states)
    M = symbol.sym_num
    # transition probability
    A=list()
    for i in range(N):
        A.append([0]*N)
    for i in range(N):
        n_i = 0
        for p in range(N):
            if (i, p) in state.transitions.keys():
                n_i += state.transitions[(i, p)]
        for j in range(N):
            if j == state.states['BEGIN']:
                A[i][j]=0.0
            elif i== state.states['END']:
                A[i][j]=0.0
            elif (i, j) in state.transitions.keys():

                A[i][j]=(state.transitions[(i, j)]+1)/(n_i+N-1)
            else:
                A[i][j] = 1.0 / (n_i + N - 1)
    # intial state
    PI=A[state.states['BEGIN']]
    # prediction
    # emission
    niD={}

    for key in symbol.emissions.keys():
        if key[0] in niD:
            niD[key[0]]+=symbol.emissions[key]
        else:
            niD[key[0]]=symbol.emissions[key]
    niD[state.states['BEGIN']]=0
    niD[state.states['END']]=0
    B={}
    # smoothing
    for i in range(N):
        B[(i, 'UNK')] = 1.0 / (niD[i] + M + 1)
        for j in range(M):
            if (i,j) in symbol.emissions.keys():
                B[(i, j)]=(symbol.emissions[(i, j)]+1)/(niD[i]+M+1)

    # predict
    ret=[]
    for query in queries.all_query:
        (prob, st)=_viterbi(A, B, N, query, symbol.symbols, state.states, PI)
        ret.append(st+[np.log(prob)])
    return ret

def _viterbi(A, B, N, Q, symbolIndex, stateIndex, PI):
    # array of Viterbi with initialization
    Vit = [{}]
    # key: states, value: path of states
    path={}
    # intialize boundary t==0
    for y in range(N):
        if Q[0] in symbolIndex.keys() and (y, symbolIndex[Q[0]]) in B.keys():
            Vit[0][y]=PI[y]*B[(y, symbolIndex[Q[0]])]
        else:
            Vit[0][y] = PI[y] * B[(y, 'UNK')]
        path[y]=[y]
    # Run Viterbi for t>0
    for t in range(1,len(Q)):
        Vit.append({})
        newPath={}

        for y in range(N):
            if Q[t] in symbolIndex.keys() and (y, symbolIndex[Q[t]]) in B.keys():
                (prob, state)=max([(Vit[t-1][y0]*A[y0][y]*B[(y, symbolIndex[Q[t]])], y0) for y0 in range(N)])
            else:
                (prob, state) = max([(Vit[t - 1][y0] * A[y0][y] * B[(y, 'UNK')], y0) for y0 in range(N)])
            Vit[t][y]=prob
            if t==len(Q)-1:
                Vit[t][y]*=A[y][stateIndex['END']]
            # append new state of current time series to path
            newPath[y]=path[state]+[y]

        path=newPath

    (prob, state)=max([(Vit[len(Q)-1][y], y) for y in range(N)])
    return (prob, [stateIndex['BEGIN']]+path[state]+[stateIndex['END']])


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k=1): # do not change the heading of the function
    # init file parser
    state = State(State_File)
    symbol = Symbol(Symbol_File)
    queries = Query(Query_File)
    queries.token_to_sym_id(symbol.symbols)
    # smoothing data from parser

    N=len(state.states)
    M = symbol.sym_num
    # transition probability
    A=list()
    for i in range(N):
        A.append([0]*N)
    for i in range(N):
        n_i = 0
        for p in range(N):
            if (i, p) in state.transitions.keys():
                n_i+=state.transitions[(i, p)]
        for j in range(N):
            if j == state.states['BEGIN']:
                A[i][j]=0.0
            elif i== state.states['END']:
                A[i][j]=0.0
            elif (i, j) in state.transitions.keys():
                A[i][j]=(state.transitions[(i, j)]+1)/(n_i+N-1)
            else:
                A[i][j]=1/(n_i+N-1)
    # intial state
    PI=A[state.states['BEGIN']]

    # prediction
    # emission
    niD={}

    for key in symbol.emissions.keys():
        if key[0] in niD:
            niD[key[0]]+=symbol.emissions[key]
        else:
            niD[key[0]]=symbol.emissions[key]
    niD[state.states['BEGIN']]=0
    niD[state.states['END']]=0
    B={}
    # smoothing
    for i in range(N):
        B[(i, 'UNK')] = 1.0 / (niD[i] + M + 1)
        for j in range(M):
            if (i,j) in symbol.emissions.keys():
                B[(i, j)]=(symbol.emissions[(i, j)]+1)/(niD[i]+M+1)

    # predict
    ret=[]
    for query in queries.all_query:
        #(prob, state)*k
        L=_viterbi_k(A, B, N, query, symbol.symbols, state.states, PI, k)
        for (prob, st) in L:
            ret.append(st+[np.log(prob)])
    return ret

# Best_K_Values(t, i) = Top K over all i,preceding_state,k (emissions[i][o_t] * m[preceding_state][k] * transition[preceding_state][i])
def _viterbi_k(A, B, N, Q, symbolIndex, stateIndex, PI, k):
    # index: time series
    # key: states
    # value: a List of k best probability
    Vit = [{}]
    # key: states, value: path of states
    path={}
    # intialize boundary t==0
    for y in range(N):
        if Q[0] in symbolIndex.keys() and (y, symbolIndex[Q[0]]) in B.keys():
            Vit[0][y]=[PI[y]*B[(y, symbolIndex[Q[0]])]]*k
        else:
            Vit[0][y]=[PI[y] * B[(y, 'UNK')]]*k
        path[y]=[[y]]*k
    # Run Viterbi for t>0
    for t in range(1,len(Q)):
        Vit.append({})
        newPath={}
        for y in range(N):
            if Q[t] in symbolIndex.keys() and (y, symbolIndex[Q[t]]) in B.keys():
                if t==1:
                    L = list(reversed(sorted([(Vit[t - 1][y0][i] * A[y0][y] * B[(y, symbolIndex[Q[t]])], y0, i) for y0 in range(N) for i in range(k)])))
                    # remove duplication
                    for t1 in L:
                        for t2 in L:
                            if t1!=t2 and t1[0]==t2[0] and t1[1]==t2[1]:
                                L.remove(t2)
                else:
                    L=list(reversed(sorted([(Vit[t-1][y0][i]*A[y0][y]*B[(y, symbolIndex[Q[t]])], y0, i) for y0 in range(N) for i in range(k)]))) if t!=len(Q)-1 else list(reversed(sorted([(Vit[t-1][y0][i]*A[y0][y]*B[(y, symbolIndex[Q[t]])]*A[y][stateIndex['END']], y0, i) for i in range(k) for y0 in range(N)])))
            else:
                if t==1:
                    L=list(reversed(sorted(set([(Vit[t-1][y0][i]*A[y0][y]*B[(y, 'UNK')], y0, i) for y0 in range(N) for i in range(k)]))))
                    # remove duplication
                    for t1 in L:
                        for t2 in L:
                            if t1 != t2 and t1[0] == t2[0] and t1[1] == t2[1]:
                                L.remove(t2)
                else:
                    L=list(reversed(sorted([(Vit[t-1][y0][i]*A[y0][y]*B[(y, 'UNK')], y0, i) for y0 in range(N) for i in range(k)]))) if t!=len(Q)-1 else list(reversed(sorted([(Vit[t-1][y0][i]*A[y0][y]*B[(y, 'UNK')]*A[y][stateIndex['END']], y0, i) for i in range(k) for y0 in range(N)])))
            Vit[t][y]=list()
            for i in range(k):
                Vit[t][y].append(L[i][0])
            # append new state of current time series to path
            if len(L)<k:
                dif=k-len(L)
                for _ in range(dif):
                    L.append((0.0, L[len(L)-1][1], -1))
            for index in range(k):
                if y not in newPath.keys():
                    newPath[y]=list()
                    newPath[y].append(path[L[index][1]][L[index][2]]+[y])
                else:
                    newPath[y].append(path[L[index][1]][L[index][2]]+[y])
        path=newPath
    ret=list(reversed(sorted([(Vit[len(Q)-1][y][i], y, i) for y in range(N) for i in range(k)])))[:k]
    ret=[(prob, [stateIndex['BEGIN']]+path[state][i]+[stateIndex['END']]) for (prob, state, i) in ret]
    return ret


def good_turing_a(transitions, N):
    ret=[[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            ret[i][j]=transitions[(i, j)]
    for i in range(N): # For all sub lists in the list (CA_ARRAY)
        if i !=N-1: # Not the last state
            N_dict=dict()
            for j in range(N):
                if ret[i][j] not in N_dict.keys():
                    N_dict[ret[i][j]]=1
                else:
                    N_dict[ret[i][j]] += 1
            sorted_dict_list=sorted(N_dict.items(),key=lambda item:item[0],reverse=False)
            c_star=sorted_dict_list[1][1]/sorted_dict_list[0][1]
            for j in range(N):
                if ret[i][j]==0:
                    ret[i][j]=c_star
            s=sum(ret[i])
            for j in range(N):
                ret[i][j]=ret[i][j]/s
    return ret

def good_turing_b(emissions, N, M):
    ret=emissions
    for i in range(N):
        N_dict=dict()
        for j in range(M):
            if ret[(i, j)] not in N_dict.keys():
                N_dict[ret[(i, j)]]=1
            else:
                N_dict[ret[(i, j)]] += 1
        sorted_dict_list=sorted(N_dict.items(),key=lambda item:item[0],reverse=False)
        if i != N - 1 and i != N - 2:
            c_star=sorted_dict_list[1][1]/sorted_dict_list[0][1]
        else:
            c_star=0
        for j in range(M):
            if ret[(i, j)]==0:
                ret[(i, j)]=c_star
        ret[(i,'UNK')]=c_star
        s=sum([ret[(i, p)] for p in range(M)])
        for j in range(M):
            if i != N - 1 and i != N - 2:
                ret[(i, j)]/=s
        if i != N - 1 and i != N - 2:
            ret[(i,'UNK')]/=s

    return ret

# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File):  # do not change the heading of the function
    # init file parser
    state = State(State_File)
    symbol = Symbol(Symbol_File)
    queries = Query(Query_File)
    queries.token_to_sym_id(symbol.symbols)
    # labels=Query_Labels(Query_Label)
    # smoothing data from parser

    N=len(state.states)
    M = symbol.sym_num

    # # transition probability
    for i in range(N):
        for j in range(N):
            if (i, j) not in state.transitions.keys():
                state.transitions[(i, j)]=0
    A = good_turing_a(state.transitions, N)
    # intial state
    PI=A[state.states['BEGIN']]
    # prediction
    # emission
    M_f={}
    for i in range(N):
        for j in range(M):
            if (i, j) in symbol.emissions.keys():
                M_f[symbol.emissions[(i, j)]]=1 if symbol.emissions[(i, j)] not in M_f.keys() else M_f[symbol.emissions[(i, j)]]+1
            else:
                M_f[0] = 1 if 0 not in M_f.keys() else M_f[0]+1
    # smoothing
    B = {}
    for i in range(N):
        for j in range(M):
            if (i, j) not in symbol.emissions.keys():
                symbol.emissions[(i, j)]=0
    B=good_turing_b(symbol.emissions, N, M)

    # predict
    ret=[]
    for query in queries.all_query:
        (prob, st)=_viterbi(A, B, N, query, symbol.symbols, state.states, PI)
        ret.append(st+[np.log(prob)])

    return ret


# ret = top_k_viterbi("toy_example/State_File", "toy_example/Symbol_File", "toy_example/Query_File", 4)
# ret2 = viterbi_algorithm("toy_example/State_File", "toy_example/Symbol_File", "toy_example/Query_File")
# ret = top_k_viterbi("dev_set/State_File", "dev_set/Symbol_File", "dev_set/Query_File", 2)
# ret2 = viterbi_algorithm("dev_set/State_File", "dev_set/Symbol_File", "dev_set/Query_File")
# ret3 = advanced_decoding("dev_set/State_File", "dev_set/Symbol_File", "dev_set/Query_File")
# print("Original Accuracy is 88.81469%")
# print('-----Output-------')
# print("top_k_viterbi")
# for i in ret:
#     print(i)
# print('----------------------------')
# print("viterbi algorithm.")
# for i in ret2:
#     print(i)
# print('----------------------------')

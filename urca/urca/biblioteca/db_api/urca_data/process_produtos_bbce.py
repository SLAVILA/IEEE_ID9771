import pandas as pd
import numpy as np
import datetime
import time

def transitivity_removal(dict_prev, dict_next):
    dict_next = dict_next.copy()
    union = {}
    for key in dict_prev:
        val = dict_prev[key]
        if val in dict_next:
            union[key] = dict_next[val]
            del dict_next[val]
        else:
            union[key] = val
    union.update(dict_next)
    return union

def _remove_hifen(produto, tipo):
    temp = produto.split(" - ")
    if len(temp) == 1:
        return produto
    if len(temp) > 2:
        # situação não prevista, múltiplos "-"
        return None
    
    tipo_dict = {'spread': 0, 'preço fixo': 1}
    temp[1] = temp[1].lower()
    if temp[1] not in tipo_dict:
        return None
    if tipo_dict[temp[1]] != tipo:
        return None
    return temp[0]

def remove_hifen(precos):
    #inic = time.perf_counter()
    precos = precos.copy()
    
    precos.produto = precos.produto.str.replace('–', '-')
    #cp = time.perf_counter()
    #print(cp - inic)
    
    temp = np.vectorize(_remove_hifen)(precos.produto, precos.tipo)
    
    #appl = time.perf_counter()
    #print(appl - cp)
    
    if any(pd.isna(np.unique(temp))):
        # foi encontrado problemas com a remoção
        print(precos[pd.isna(temp)])
        assert(not any(pd.isna(np.unique(temp))))
    precos.produto = temp
    
    #fim = time.perf_counter()
    #print(fim - appl)
    return precos

def split_products(precos, corrections=None):
    precos = precos.copy()
    if corrections:
        precos['produto'] = precos['produto'].replace(corrections)
    partes = {}
        
    def f(p):
        temp = p.split()
        if len(temp) not in partes:
            partes[len(temp)] = {tuple(temp)}
        else:
            partes[len(temp)].add(tuple(temp))
    precos.produto.apply(f)
    return {n: np.array(list(p)) for n, p in partes.items()}

def show_part(part):
    for i in range(part.shape[1]):
        print(np.unique(part[:, i]))
    print()

def p3_corrections(precos):
    part = split_products(precos)[3]
    return {" ".join(x): " ".join([x[0], x[1], "MEN", x[2]]) for x in part}

def to_date(code):
    try:
        month2date = {
            "jan": 1, "fev":  2, "mar":  3, "abr":  4,
            "mai": 5, "jun":  6, "jul":  7, "ago":  8,
            "set": 9, "out": 10, "nov": 11, "dez":  12
        }

        month, year = code.split("/")
        y = 2000+int(year)
        m = month2date[month.lower()]
        return datetime.datetime(y, m, 1)
    except:
        return None

def verify_all_months(parts, loc):
    months = np.vectorize(to_date)(parts[:, loc])
    return parts[:, loc][pd.isna(months)]

def p4_corrections(precos):
    p3_corr = p3_corrections(precos)
    part = split_products(precos, p3_corr)[4]
    p4 = np.array(list(part))
    p4 = p4[p4[:, 2] != 'MEN']
    p4_corr = {" ".join(i): " ".join([i[0], i[1], "OTR", i[2], i[3]]) for i in p4}
    return transitivity_removal(p3_corr, p4_corr)

def p5_corrections(precos):
    p4_corr = p4_corrections(precos)
    part = split_products(remove_hifen(precos), p4_corr)[5]
    p5 = part[part[:, 2] == "MEN"]
    p5_new = p5.copy()
    p5_new[:, 2] = "OTR"
    p5_corr = {" ".join(p5[i]): " ".join(p5_new[i]) for i in range(len(p5))}
    return transitivity_removal(p4_corr, p5_corr)

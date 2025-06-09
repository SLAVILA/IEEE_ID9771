import time

def get_closing(df, product_col, data_col, time_col):
    index = df.groupby([product_col, data_col])[time_col].idxmax()
    return df.loc[index]
# get_closing(preco_bbce_men_preco_fixo, "produto", "data", "data_hora")

def get_maturidade(raw_df, product_col, data_col, copy=False):
    raw_df["M"] = raw_df.apply(lambda x: (x["expiracao"].month + x["expiracao"].year*12) - (x["data"].month + x["data"].year*12)-1, axis=1)
    
    H_dict = raw_df.groupby(product_col)[data_col].count().to_dict()
    
    final_df = raw_df.copy() if copy else raw_df
    final_df['H'] = final_df[product_col].apply(lambda x: H_dict[x])
    
    def helper(row):
        row = row.sort_values(data_col)
        row["h"] = range(len(row)-1, -1, -1)
        row["h_cresc"] = range(len(row))
        return row
    final_df = final_df.groupby(product_col, group_keys=False).apply(helper)  
    return final_df
# get_maturidade(preco_bbce_men_preco_fixo, "produto", "data")

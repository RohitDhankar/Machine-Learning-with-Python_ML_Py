import pandas as pd
# prop calc
buy = 10 # if buy > 5 - then 10 Bonus
rent = 5 # if rent > 8 - then Add 10% to Total Base Price 
short_trm = 2.5

# dict_calc
dict_calc = {}
dict_calc["ls_buy"] = []
dict_calc["ls_rent"] = []
dict_calc["ls_shrt_trm"] = []

def calc_input(p_rule = " "):
    if p_rule == "buy":
        dict_calc["ls_buy"].append(10)
        return dict_calc
    elif p_rule == "rent":
        dict_calc["ls_rent"].append(5)
        return dict_calc
    elif p_rule == "short_trm":
        dict_calc["ls_shrt_trm"].append(2.5)
        return dict_calc

userInput = input()#(rent)    
dict_calc = calc_input(userInput)
print("--dict_calc-AA-",dict_calc)


def calc_rules(dict_calc):
    ls_buy = []
    ls_rent = []
    ls_a = []
    #userInput = input()#(rent)    
    #dict_calc = calc_input(userInput)
    #print("--dict_calc-AA-",dict_calc)
    #if dict_calc:
        
    df = pd.DataFrame([dict_calc],columns=dict_calc.keys())
    print(df)
    #dfA = pd.DataFrame()
    dfA = pd.concat([dfA, df], axis =0).reset_index()

    #dfA = df.append(dict_calc, ignore_index=True)
    print(dfA)
    print("--BBBB----")
    return dfA

dfA = calc_rules(dict_calc)
print("--AAAAA----")
print(dfA)





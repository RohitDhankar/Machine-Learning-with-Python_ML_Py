import pandas as pd
# prop calc
"""
buy = 10 # if buy > 5 - then 10 Bonus
rent = 5 # if rent > 8 - then Add 10% to Total Base Price 
short_trm = 2.5
"""

# dict_calc
dict_calc = {}
dict_calc["ls_buy"] = []
dict_calc["ls_rent"] = []
dict_calc["ls_shrt_trm"] = []

def calc_input(p_rule = " "):
    # ls_buy = []
    # ls_rent = []
    # ls_a = []

    counter_n = 0 
    if p_rule:
        counter_n +=1
        #print("counter_n ---" ,counter_n)
    
    if p_rule == "buy":
        buy = 10
        #ls_buy.append(buy)
    else:
        buy = 0
    if p_rule == "rent":
        rent = 5
        #ls_rent.append(rent)
    else:
        rent = 0 
    if p_rule == "short_trm":
        short = 2.5
    else:
        short = 0
    return buy , rent , short ,counter_n

userInput = input()#(rent)    
#ls_buy , ls_rent , short = calc_input(userInput)
buy , rent , short , counter_n = calc_input(userInput)
print("buy----",buy)
#print("--dict_calc-AA-",buy , rent , short , counter_n)


def calc_rules(buy , rent , short , counter_n):

    ls_buy = []
    ls_rent = []
    ls_a = []
    ls_buy.append(buy)
    ls_rent.append(rent)
    #
    print("ls_buy----",ls_buy)
    #print(ls_rent)
    #dfA = pd.DataFrame({'buy':ls_buy,'rent':ls_rent}) 
    #dfA = pd.DataFrame({'buy':ls_buy}) 
    dfA = pd.DataFrame(columns=["buy","rent","short_trm"]) 
    #for k in range(counter_n):
    dfA.loc[counter_n+1] = [ls_buy]
    #print(dfA)
    #dfA = pd.DataFrame({'buy':ls_buy,'rent':ls_rent})  
    # zipped_lists = zip(ls_buy,ls_rent) # ZIpping Ok if we are returning DICT 
    # print(zipped_lists)
    # #    
    # df = pd.DataFrame([dict_calc],columns=dict_calc.keys())
    # print(df)
    # #dfA = pd.DataFrame()
    # dfA = pd.concat([dfA, df], axis =0).reset_index()

    # #dfA = df.append(dict_calc, ignore_index=True)
    # print(dfA)
    # print("--BBBB----")
    return dfA

dfA = calc_rules(buy , rent , short , counter_n)
print(dfA)
dfA.to_csv('my_csv.csv', mode='a', header=False)

# dfB = dfA
# def concatDF(dfB):
#     dfA = pd.concat([dfA, df], axis =0).reset_index()


# dfA = calc_rules(dict_calc)
# print("--AAAAA----")
# print(dfA)




#print("ls_buy----",ls_buy)
    #print(ls_rent)

    # dict_calc["ls_buy"].append(str(buy))
    # dict_calc["ls_rent"].append(str(rent))
    # dict_calc["ls_shrt_trm"].append(str(short))
    # print("dict_calc--------",dict_calc)

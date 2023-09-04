import pandas as pd
# Argparse 
import argparse
parser = argparse.ArgumentParser(description='Code for Calculator.')
parser.add_argument('--buyCount', help='Buy count Integer value.- default=6', default='6')
parser.add_argument('--rentCount', help='Rent count Integer value.- default=2', default='2')
parser.add_argument('--shortCount', help='Short Term count Integer value.- default=1', default='1')
args = parser.parse_args()

"""
Calculation Rules 
buy = 10 # if buy > 5 - then Pound 10 Bonus
rent = 5 # whenever rent leads > 8 - then Add 10% to Total Base Price 
short_trm = 2.5
"""

buyC = args.buyCount
print(buyC)
rentC = args.rentCount
shortC = args.shortCount


# def calc_indlInput(p_rule = " "):
#     counter_n = 0 
#     if p_rule:
#         counter_n +=1
#         print("counter_n ---" ,counter_n)
#     if p_rule == "buy":
#         buy = 10
#     else:
#         buy = 0
#     if p_rule == "rent":
#         rent = 5
#     else:
#         rent = 0 
#     if p_rule == "shrt":
#         short = 2.5
#     else:
#         short = 0
#     return buy , rent , short ,counter_n

#userInput = input()#Alternative of ARGPARSE - here we provide Individual inputs like - buy , rent ...
#buy , rent , short , counter_n = calc_indlInput(userInput)

# buy , rent , short , counter_n = calc_input(userInput)
# print("buy , rent , short , counter_n---", buy , rent , short , counter_n)

# dict_calc = {}
# dict_calc["ls_buy"] = []
# dict_calc["ls_rent"] = []
# dict_calc["ls_shrt_trm"] = []

ls_buy = []
ls_rent = []
ls_short_trm = []
dfA = pd.DataFrame(columns=["buy","rent","short_trm"]) 

def calc_rules(buyC , rentC , shortC):
    counter_n = 0 
    ls_buy.append(buyC)
    ls_rent.append(rentC)
    ls_short_trm.append(shortC)
    # while counter_n < 6:
    #     counter_n +=1
    #     print("counter_n ---" ,counter_n)
    #dfA.loc[counter_n+1] = [ls_buy[0]] + [ls_rent[0]] + [ls_short_trm[0]]
    dfA.loc[1] = [ls_buy[0]] + [ls_rent[0]] + [ls_short_trm[0]]
    return dfA

dfA = calc_rules(buyC , rentC , shortC)
#print("dfA------",dfA)
dfA.to_csv('my_csv.csv', mode='a', header=False)
#
dfCSV = pd.read_csv('my_csv.csv')
print("   "*90)
print("dfCSV------BELOW-----------------------",type(dfCSV))
print("dfCSV------",print(dfCSV))
#
def bonusCalc(dfCSV):
    dfCSV.columns = ['temp','buy','rent','short']
    print(dfCSV)
    buySum = dfCSV["buy"].sum()
    div_varbuySum = buySum/50 
    if div_varbuySum in list(range(1,100)):
        print("div_varbuySum-----additional_bonus = True-----")
        buy_bonus = 1
    else:
        buy_bonus = 0

    rentSum = dfCSV["rent"].sum()
    div_rentSum = rentSum/40
    print(div_rentSum)
    if div_rentSum in list(range(1,100)):
        print("div_rentSum---additional_bonus = True-----")
        rent_bonus = 1
    else:
        rent_bonus = 0
    shortSum = dfCSV["short"].sum()
    print("buySum-------",buySum)
    print("rentSum------",rentSum)
    print("shortSum-----",shortSum)

bonusCalc(dfCSV)





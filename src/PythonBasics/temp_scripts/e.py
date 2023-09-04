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

ls_buy = []
ls_rent = []
ls_short_trm = []
dfA = pd.DataFrame(columns=["buy","rent","short_trm"]) 

def calc_rules(buyC , rentC , shortC):
    if int(buyC) <= 5:
        ls_buy.append(int(buyC)*10)
    else:
        ls_buy.append((int(buyC)*10)+10)
    if int(rentC) <= 8:
        ls_rent.append(int(rentC)*5)
        counter_rent = 0
    else:
        ls_rent.append(int(rentC)*5)
        counter_rent = 1 
        
    
    ls_short_trm.append(float(shortC)*2.5)
    dfA.loc[1] = [ls_buy[0]] + [ls_rent[0]] + [ls_short_trm[0]]
    return dfA , counter_rent

dfA , counter_rent = calc_rules(buyC , rentC , shortC)
#print("dfA------",dfA)
dfA.to_csv('my_csv.csv', mode='a', header=False)
dfCSV = pd.read_csv('my_csv.csv')
print("   "*90)
#print("dfCSV------BELOW-----------------------",type(dfCSV))
#print("dfCSV------",print(dfCSV))
#
def bonusCalc(dfCSV , counter_rent):
    dfCSV.columns = ['temp','buy','rent','short']
    print(dfCSV)
    buySum = dfCSV["buy"].sum()
    div_varbuySum = buySum/50 
    if 50 < buySum < 100:
        print("----BuySum == 60 -----")
    if div_varbuySum in list(range(1,100)):
        print("div_varbuySum-----additional_bonus = True-----")
        buy_bonus = 1
    else:
        buy_bonus = 0

    rentSum = dfCSV["rent"].sum()
    print("----VAL----counter_rent----",counter_rent)

    shortSum = dfCSV["short"].sum()
    total = buySum + rentSum + shortSum

    print("buySum-------",buySum)
    print("rentSum------",rentSum)
    print("shortSum-----",shortSum)
    #print("total ----", total)

    return total


total = bonusCalc(dfCSV)
print(total)





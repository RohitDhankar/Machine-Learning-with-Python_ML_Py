import pandas as pd
# Argparse 
import argparse
parser = argparse.ArgumentParser(description='Code for Calculator.')
parser.add_argument('--buyCount', help='Buy count Integer value.- default=6', default='1')
parser.add_argument('--rentCount', help='Rent count Integer value.- default=2', default='10')
parser.add_argument('--shortCount', help='Short Term count Integer value.- default=1', default='0')
args = parser.parse_args()

"""
Calculation Rules 
buy = 10 # if buy > 5 - then Pound 10 Bonus
rent = 5 # whenever rent leads > 8 - then Add 10% to Total Base Price 
short_trm = 2.5
"""

buyC = args.buyCount
rentC = args.rentCount
shortC = args.shortCount
print(shortC)

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
print("   "*90)
print("dfA------",dfA)
# Write CSV Column Labels - then keep appending data 
dfA.to_csv('my_csv.csv', mode='a', header=False)
dfCSV = pd.read_csv('my_csv.csv')
print("   "*90)
#

dfB = pd.DataFrame(columns=["buy","rent","short_trm","row_sum"]) 
def rowCalc(buyC , rentC , shortC):
    ls_b = ["A"]
    ls_r = ["B"]
    ls_sh = ["C"]
    # print("     "*90)
    # print(dfB)
    # print("     "*90)
    #
    if int(buyC) <= 5:
        b2 = int(buyC)*10
        #ls_b.append(int(buyC)*10)
    else:
        b2 = (int(buyC)*10)+10
        #ls_b.append((int(buyC)*10)+10)
    if int(rentC) <= 8:
        r2 = int(rentC)*5
        #ls_r.append(int(rentC)*5)
        counter_rent = 0
    else:
        r2 = int(rentC)*5
        #ls_r.append(int(rentC)*5)
        counter_rent = 1 
    sh2 = float(shortC)*2.5
    #ls_sh.append(float(shortC)*2.5)
    if counter_rent == 1:
        rowSum = sh2 + r2 + b2
        rowSum = (rowSum*10)/100 + rowSum
        print("---AA---rowSum-------",rowSum)
    else:
        rowSum = sh2 + r2 + b2 
        print("---BB---rowSum-------",rowSum)
    print("---AA---b2-------------",b2)
    print("---AA---r2-------------",r2)
    print("---FINAL---rowSum-------",rowSum)
    dfB.loc[1] = [b2] + [r2] + [sh2] + [rowSum]
    #dfB.loc[1] = [ls_b[0]] + [ls_r[0]] + [ls_sh[0]]
    print("     "*90)
    print("----dfB-------------------",dfB)
    print("     "*90)
    # Write CSV Column Labels - then keep appending data 
    dfB.to_csv('totals.csv', mode='a', header=False)
    dfBCSV = pd.read_csv('totals.csv')
    
    

    return rowSum


def bonusCalc(dfCSV , counter_rent):
    dfCSV.columns = ['temp','buy','rent','short']
    print("dfCSV------BELOW-----------------------",print(dfCSV))
    buySum = dfCSV["buy"].sum()
    rentSum = dfCSV["rent"].sum()
    print("----VAL----counter_rent----",counter_rent)
    shortSum = dfCSV["short"].sum()
    total = "dummyStr"
    if counter_rent == 1:
        totalA = buySum + rentSum + shortSum
        print("----Total_A-----",totalA)
        if totalA > 0 :
            ten_p = 100/totalA*10
            totalB = ten_p + totalA
            print("----Total_B-----",totalB)
    else:
        total = buySum + rentSum + shortSum

    print("buySum-------",buySum)
    print("rentSum------",rentSum)
    print("shortSum-----",shortSum)

    return total

total = bonusCalc(dfCSV , counter_rent)
print("total ----", total)

rowSum = rowCalc(buyC , rentC , shortC)
print("Total --- rowSum ----", rowSum)







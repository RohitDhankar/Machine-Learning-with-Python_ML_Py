"""
# Test Data 
python calc.py --buyCount 6 --rentCount 2 --shortCount 1
python calc.py --buyCount 1 --rentCount 10 --shortCount 0
"""

import pandas as pd
# Argparse - pass in arguments from terminal 
import argparse
parser = argparse.ArgumentParser(description='Code for Calculator.')
parser.add_argument('--buyCount', help='Buy count Integer value.- default=6', default='6')
parser.add_argument('--rentCount', help='Rent count Integer value.- default=2', default='2')
parser.add_argument('--shortCount', help='Short Term count Integer value.- default=1', default='1')
args = parser.parse_args()
buy = int(args.buyCount)
rent = int(args.rentCount)
short = int(args.shortCount)

dfB = pd.DataFrame(columns=["buy","rent","short_trm","row_sum"]) 
def rowCalc(buy,rent,short):
    if buy <= 5:
        b2 = buy*10
    else:
        b2 = (buy*10)+10
    if rent <= 8:
        r2 = rent*5
        cntR = 0
    else:
        r2 = rent*5
        cntR = 1 
    sh2 = short*2.5
    if cntR == 1:
        rowSum = sh2 + r2 + b2
        rowSum = rowSum + (rowSum*10)/100
    else:
        rowSum = sh2 + r2 + b2 
    dfB.loc[1] = [b2] + [r2] + [sh2] + [rowSum]
    dfB.to_csv('totals.csv', mode='a', header=False)
    dfBCSV = pd.read_csv('totals.csv')
    print("Total -rowSum ----", rowSum)
    return rowSum

rowSum = rowCalc(buy , rent , short)








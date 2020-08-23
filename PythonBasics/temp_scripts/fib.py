#Fibonaci Sequence ( Series )
import argparse
parser = argparse.ArgumentParser(description='Code for Fibonaci Sequence.')
parser.add_argument('--tCnt', help='Series with LENGTH / Required terms in the Series.- default=10', default='10')
args = parser.parse_args()
tCnt = int(args.tCnt)

fibls=[]

def fib1(tCnt):
    el_1 =0   #1st ele Fib Seq
    el_2 =1   #2nd ele Fib Seq
    a = int(tCnt)
    if a<=0:
        print(" U entered a Negative Num or ZERO ")
    else:
        print(el_1,el_2) # Static intial Values - 0 and 1 
        for k in range(2,a): # START - 2 , STOP -a 
            next=el_1+el_2                          
            fibls.append(next)
            el_1=el_2
            el_2=next
        return fibls

fibls = fib1(tCnt)
print(fibls)

#

def fib2():
    a, b = 0,1
    while b < 10:
        print(b) # Fib Series 
        a, b = b, a+b # Switch Places and Sum 

fib2()

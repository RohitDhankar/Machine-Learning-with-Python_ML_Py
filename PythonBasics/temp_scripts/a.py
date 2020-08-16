def boardGame(X,Y):
    result = -404
    X1 = X + 1 
    X2 = X1 * X1
    X3 = X2 + 1 

    if Y == X3:
        #print("Result -- Y == X3--",X3)
        result = 3
    else:
        #print("Result -- Y != X3--",X3)
        print(X3)
        result = 0 
    return result
    
#INPUT [uncomment & modify if required]
temp = input().split() # 2 SPACE 10

X = int(temp[0])
Y = int(temp[1])

#OUTPUT [uncommet & modify if required]
print(boardGame(X,Y))
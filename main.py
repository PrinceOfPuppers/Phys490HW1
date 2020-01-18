import sys
import numpy as np
import json

def loadData(sysArgs):
    rawDataPath=sys.argv[1]
    hypParamPath=sys.argv[2]
    outDataPath=sys.argv[1][0:-2]+"out"
    hypParams=json.loads(open(hypParamPath).read())
    rawData=open(rawDataPath)
    inData=[]
    for line in rawData:
        #formats line into a list of floats
        line=list(map(float,line.split()))
        inData.append(line)

    return(inData,hypParams,outDataPath)

def createOutFile(wLinReg,wSGD,outPath):
    outData=open(outPath,"w+")
    for component in wLinReg:
        rounded=round(component,3)
        outData.write(str(rounded)+"\n")
    
    outData.write("\n")
    for component in wSGD:
        rounded=round(component,3)
        outData.write(str(rounded)+"\n")
    
    print(">> .out file created at {}\n".format(outPath))

def linReg(inData):
    phiM=[]
    ys=[]

    #populates phiM and y
    for line in inData:
        #x first element will always be 1.0
        x=[1.0]+line[0:-1]

        phiM.append(x)
        ys.append(float(line[-1]))
    
    #converts both to np array
    phiM=np.asarray(phiM,float)
    ys=np.asarray(ys,float)

    #gets transpose as well as a copy of to mutate into w
    phiMT=np.copy(phiM).transpose()

    #preforms matrix operations
    a=np.matmul(phiMT,phiM,dtype=float)
    b=np.linalg.inv(a)
    c=np.matmul(b,phiMT,dtype=float)
    w=np.matmul(c,ys,dtype=float)

    return(w)

def stocGradDecent(inData,hypParams):
    alpha=hypParams["learning rate"]
    numIter=hypParams["num iter"]

    #load xs and ys
    xs=[]
    ys=[]
    for line in inData:
        #x first element will always be 1.0
        x=[1.0]+line[0:-1]
        x=np.array(x,dtype=float)

        y=float(line[-1])

        xs.append(x)
        ys.append(y)

    w=np.array([0]*len(xs[0]),dtype=float)
    #loops through dataset numIter times
    for i in range(0,numIter):
        for j in range(0,len(xs)):
            x=xs[j]
            y=ys[j]
            w+=alpha*np.dot((y-np.dot(w,x)),x)
    return(w)


if __name__ == "__main__":
    inData,hypParams,outPath=loadData(sys.argv)
    print("\n>> Data Loaded")

    wLinReg=linReg(inData)
    print(">> linear Regression Calculated")

    wSGD=stocGradDecent(inData,hypParams)
    print(">> SGD Calculated with Learning Rate {} and {} Iterations".format(hypParams["learning rate"],hypParams["num iter"]))

    createOutFile(wLinReg,wSGD,outPath)
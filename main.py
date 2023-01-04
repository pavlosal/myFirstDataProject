"""
import numpy as np


arr =np.array([1,2,3,4,5])


arr=np.array(10)
print(arr)
#print(type(arr))

arr=np.array(10)
arr1= np.array([[[1,2,3,19],[3,4,5,15]],[[44,6,9,64],[10,11,12,999]]])
arr2 =np.array([1,2,3,4,5])
#print(arr.ndim, arr1.ndim, arr2.ndim)
print(arr1[1,0,-1])

#arr=np.array([1,2,8,4,5,6,185,8,9,1625,1])

arr= np.array([[4,7,9,3,5,7],[1,5,8,8,2,5]])

print(arr[0,1:10:2])

arr2 =np.array([1,2,3,4,5])
copy=arr2.copy()

copy[3]=8

print(arr2)
print(copy)

arr2 =np.array([1,2,3,4,5])
skata=arr2.view()

skata[3]=8

print(arr2)
print(skata)



# kartela numphy shape & Reshape

arr=np.array([1,2,3,4,5,6,7,8,9])
view=arr.view()

print(arr)
print()
print(arr.reshape(9,1))

arr=np.array([[[1,2,3,19],[3,4,5,15]],[[44,6,9,64],[10,11,12,999]]])
view=arr.view()

print(arr)
print()
print(arr.shape)


arr=np.array([1,2,9,4,5,3,7,8,6,0])
#arrList=np.array_split(arr,5)

#for array in arrList:
#print(np.where(arr == 2))

print(np.where(arr % 2 == 0))


arr=np.array([1,2,9,4,5,3,7,8,6,0])

print(np.sort(arr))

arr1=np.array([1,2,3,4,5,6])
arr2=np.array([6,5,4,3,2,1])
arr3=np.array([5,6,7,8,1,2])

print(np.subtract(arr1,arr2))
print(np.add(arr1,arr3))
print(np.multiply(arr2,arr3))
print(np.divide(arr3,arr1))
print(np.power(arr2,arr1))
print(np.absolute(np.subtract(arr1,arr2)))
print(np.mod(arr2,arr1))


import numpy as np

arr=np.array([1,4,1.32,7.99,6.521456987,100])
arr1=np.array([12365478.32])
arr2=np.array([1,2,5,7,3.72,16.789])
arr3=np.array([1,2,5,7,3,16])
arr4=np.array([144,64,736,100])
arr5=np.array([np.pi/2, np.pi/6, np.pi/4])
arr6=np.array([30,45,60,90,180])
print(np.trunc(arr))
print(np.fix(arr))
print(np.around(arr,4))
print(np.ceil(arr))
print(np.floor(arr))
print(np.log(arr))
print(np.log10(arr))
print(np.log2(arr))
print(np.sum([arr,arr2]))
print(np.sum([arr,arr2],axis=1))
print(np.cumsum(arr2))
print(np.prod([arr,arr2])) # polaplasiazei ta panta
print(np.prod([arr,arr2],axis=1))
print(np.cumprod(arr3))
print(np.lcm.reduce(arr3))  #elachisto koino polaplasio
print(np.gcd.reduce(arr4))
print(np.around(np.sin(arr5),4))  #xrisimopoioyn panta rad, gia moires delei metatroph
print(np.around(np.cos(arr5),8))
print(np.deg2rad(arr6))
print(np.rad2deg(arr5))
print(np.hypot(8,6))



import pandas as pd

x=[23,48,19,]
myfirstseries=pd.Series(x)
print(myfirstseries)
"""

import pandas as pd
"""
data={ "ahjdgf":['emma','mpampis',3,4],
       "ahdjkldff":['mitsos','skouliki','vlakas',2]}
dataframe1=pd.DataFrame(data)
print(dataframe1)

import numpy as np
data= { "colours":["red","green",np.nan,"purple"],
        "countries":["Maroco","Portugal","No idea",np.nan]}
data1= { "colours":["red","green",np.nan,"purple"],
        "Numbers":[12,np.nan,np.nan,21]}
framedata1=pd.DataFrame(data, index=["proti","deyterh","triti","tetarti"])
dataframe1=pd.DataFrame(data1, index=["proti","deyterh","triti","tetarti"])
df2=dataframe1.interpolate(method='linear',limit_direction='forward')
df1=framedata1.replace(to_replace="purple",value="pink")
dataframe1.dropna(inplace=True)
#print(framedata1[data])
print(framedata1["colours"])
print(framedata1["countries"])
print(framedata1.loc["triti"])
print(framedata1.iloc[2])
print(framedata1.isnull())
#print(framedata1["countries"].fillna("nothing",inplace=True))
framedata1["countries"].fillna("No Country",inplace=True)
framedata1["colours"].fillna("No Colour",inplace=True)
print(framedata1)
print(df1)
print(df2)
print(dataframe1)

import pandas as pd

s=pd.Series(['workearly','elearning','python'])
for index, value in s.items():
    print("Index: {index}, Value: {value}")

import pandas as pd

data= { "colours":["red","green","purple"],
        "Numbers":[12,18,21]}

dataframe1=pd.DataFrame(data, index=["a","b","c"])
for i,j in dataframe1.iterrows():
    print(i,j)
    print()
columns=list(dataframe1)
for i in columns:
    print(dataframe1[i][2])


import pandas as pd

df=pd.read_csv("finance_liquor_sales.csv")
#print(df.head(10))
#print(df.tail(5))
#print(df.info())
print(df.shape)


import pandas as pd

df=pd.read_csv("finance_liquor_sales.csv")
mean= df.mean(numeric_only=True)
median = df.median(numeric_only= True)
maxv= df.max(numeric_only= True)
summary= df.describe()
print(summary)


import pandas as pd
#import numpy as  np

df= pd.read_csv("finance_liquor_sales.csv")

cn=df.groupby('category_name')
cn2=df.groupby(['category_name','city'])

ng=df.groupby('vendor_name')


#print(cn2.first())
#print(cn.aggregate(np.sum))
#print(cn2.agg({'bottles_sold':'sum','sale_dollars':'mean'}))
print(ng.filter(lambda x: len(x)>=20))


import pandas as pd

d1={'Name':['Mary','John','Nick','Bob'],'Age':[21,22,24,45],'Position':['Data Analyst','Trainee','QA Tester','IT']}
d2={'Name':['Stavros','Tomas','Jenny','George'],'Age':[42,25,32,19], 'Position':['IT','Data Analyst','Consultant','IT']}

df1=pd.DataFrame(d1, index=[0,1,2,3])
df2=pd.DataFrame(d2, index=[4,5,6,7])
result= pd.concat([df1,df2])
print(result)


import pandas as pd


d1={'key':['a','b','c','d'],'Age':[21,22,24,45],'Position':['Data Analyst','Trainee','QA Tester','IT']}
d2={'key':['a','b','c','d'],'Age':[42,25,32,19], 'Position':['IT','Data Analyst','Consultant','IT']}

df1=pd.DataFrame(d1)
df2=pd.DataFrame(d2)
result=pd.merge(df1,df2, on='key')
print(result)


d1={'Name':['Mary','John','Nick','Bob'],'Age':[21,22,24,45]}
d2={'Position':['IT','Data Analyst','Consultant','IT'],'Years_of_experience':[5,3,7,23]}

df1=pd.DataFrame(d1, index=[0,1,2,3])
df2=pd.DataFrame(d2, index=[0,1,2,3])

result=df1.join(df2, how='inner')
print(result)


import pandas as pd

L=[5,10,15,20,25]
ds=pd.Series(L)
print(ds)


import pandas as pd

d={'col1':[1,2,3,4,7,11],'col2':[4,5,6,9,5,0],'col3':[7,5,8,12,1,11]}

df=pd.DataFrame(d)
s1=df.iloc[:,0]
print('Cirst Column as a Series')
print(s1)
print(type(s1))

import pandas as pd
df=pd.read_csv('data.csv')
for i,j in df.iterrows():
    print(i, j)



import pandas as pd
import numpy as np

data= pd.read_csv('1.supermarket.csv')

#print(data.head())
#print(data.tail())
#print("\nShape of dataset:", data.shape)
#print(data.info())

print(data.columns)

x= data.groupby('item_name')

x=x.sum()

print(x.head())


import matplotlib.pyplot as plt

#plt.plot([0,10], [0,300], 'o')
plt.plot([0,1,2,3,4,5,6,7,8,9],[3,1,5,9,1,7,2,9,5,15],[1,2,3,4,5,6,7,8,9,10], marker='o',ls='dotted')

plt.title("Titlos")
plt.xlabel('X-Axis')
plt.ylabel('y-Axis')
plt.grid()

plt.show()

import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.plot([1,2,3,4],[4,3,2,1],marker='o',ls='dotted')
plt.grid()

plt.subplot(2,1,2)
plt.plot([1,5,9,3],[6,7,3,8])

plt.show()


import matplotlib.pyplot as plt
import numpy as np

x=np.array([15,63,85,16,5,51,61,71,21,41,91])

y=np.array([3,4,5,7,1,9,8,2,6,4,12])

plt.scatter(x,y)

x=np.array([16,64,86,16,5,162,161,72,22,41,91])

y=np.array([6,7,22,4,6,1,14,16,8,4.2,3])

plt.scatter(x,y)

plt.show()


import matplotlib.pyplot as plt
import numpy as np

x=np.array(["A","B","C","D"])

y=np.array([6,1,4,0.2])

plt.bar(x,y)

plt.show()


import matplotlib.pyplot as plt
import numpy as np

mylabels=np.array(["apples","oranges","tomatos","strawberies"])

x=np.array([2,6,5,7])

plt.pie(x,labels=mylabels)
plt.legend()

plt.show()


import matplotlib.pyplot as plt

age= [10,20,30,40,50,60,70,80,90,100]
cardiac_cases= [5,15,20,40,55,55,70,80,90,95]
survival_chances= [99,99,90,90,80,75,60,50,30,25]

plt.xlabel("Age")
plt.ylabel("Percentage")

plt.plot(age, cardiac_cases, color='black', linewidth=2,label="Cardiac Cases",marker='o', markerfacecolor='red',markersize=12)
plt.plot(age, survival_chances, color='yellow',linewidth=2,label="Survival Chances", marker='o', markerfacecolor='green', markersize=12)
plt.legend(loc='lower right', ncol=1)

plt.show()


import numpy as np
import matplotlib.pyplot as plt

products=np.array([["Apple","Orange"],["Beef","Chicken"],["Candy","Chocolate"],["Fish","Bread"],["Egggs","Bacon"]])

random=np.random.randint(2,size=5)

choices =[]

counter =0

for product in products:
    choices.append(product[random[counter]])
    counter +=1

percentages= []

for i in range(4):
    percentages.append(np.random.randint(25))
percentages.append(100- np.sum(percentages))

print(percentages)

plt.pie(percentages, labels= choices)
plt.legend(loc='lower right', ncol=1)

plt.show()


import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv('1.supermarket.csv')

q= data.groupby('item_name').quantity.sum()

plt.bar(q.index,q, color=['orange','purple','yellow','red','black','blue','green'])
plt.xlabel('Items')
plt.xticks(rotation=10)
plt.ylabel('Number of times Ordered')
plt.title('Most Ordered Supermarket\'s Items')
plt.show()



import requests
from bs4 import BeautifulSoup


url= "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt=requests.get(url).text
s=BeautifulSoup(url_txt,"html.parser")
#print(s.prettify)

#print(s.title)
#print(s.title.string)

#tag=s.find_all('a')
#table=s.find_all('table')
#print(tag)

my_table=s.find('table',class_='wikitable sortable plainrowheaders')
table_links=my_table.find_all('a',href= True)
#print(table_links)

actors = []
for links in table_links:
    actors.append(links.get('title'))
print(actors)
"""


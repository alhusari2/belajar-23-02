import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('GammaMart_Customers.csv')
df.head(10)
print(df.info())

import matplotlib.pyplot as plt
fig = plt.figure()
ax=fig.add_axes ([0,0,1,1])
ax.axis('equal')
ax.pie(cust,labels = ['Males','Females'], autopct='%1.0f%%')
plt.show()
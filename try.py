class Kalkulator:
    """contoh kelas kalkulator sederhana. anggap kelas ini tidak boleh diubah!"""
 
    def __init__(self, nilai=0):
        self.nilai = nilai
 
    def tambah_angka(self, angka1, angka2):
        self.nilai = angka1 + angka2
        if self.nilai > 9:  # kalkulator sederhana hanya memroses sampai 9
            print('kalkulator sederhana melebihi batas angka: {}'.format(self.nilai))
        return self.nilai

class KalkulatorKali(Kalkulator):
    """contoh mewarisi kelas kalkulator sederhana"""
 
    def kali_angka(self, angka1, angka2):
        self.nilai = angka1 * angka2
        return self.nilai
 
    def tambah_angka(self, angka1, angka2):
        self.nilai = angka1 + angka2
        return self.nilai

kk = KalkulatorKali()
 
b = kk.tambah_angka(5, 6)  # fitur tambah_angka yang dipanggil milik KalkulatorKali
print(b)


import matplotlib.pyplot as plt
import numpy as np
def fa (net, teta):
    if (net<-teta) :
       return -1;
    elif (net>teta):
       return 1;
    else:
       return 0;

X=np.array ([[1,1], [1,-1], [-1,1],[-1,-1]])
bdata=len(X);
teta=0.1
print ("Banyak data: ",bdata)
bfeat=len (X [0,:])
print ("Banyak Feature: ",bfeat)
T=np.array([1,-1,-1,-1])
print (X)
print (T)
alfa=float (input ("Learning rate:"))

w=np.zeros (2)
b=np.random.rand ()
for i in range (2):
   w[i]=np.random.rand ()
print ("Bobot Awal:",w)
print ("Bias Awal:",b)
berhenti=False
ItMax=100
it=0
while (berhenti==False and it<ItMax):
    berhenti = True
    faktif=np.zeros (bdata)
    for i in range (bdata):
        net=b
        for j in range (bfeat):
          net=net+w[j] *X[i,j]
        faktif[i]=fa (net, teta)
        if faktif[i]!=T[i]:
          berhenti = False
        for j in range (bfeat):
          w[j] =w[j]+alfa*T[i]*X[i,j]
        b=b+alfa*T[i]
        it=it+1
print ("Jumlah iterasi=",it)
print ("Bobot Akhir:",w)
print ("Bias Akhir:",b)
plt.scatter (X[:,0],T,s=100, c='b', marker='o')
plt.scatter (X[ :,0],faktif, s=100, c='y', marker='*')
plt.xlabel ("Feature 1 (x1)")
plt.ylabel("Keluaran Y/T")
plt.title ("Perbandingan Keluaran Jaringan Perceptron dan Target");
plt.show ()

inp = int(input('Masukkan Sampel => '))
table = []
for i in range(1, inp+1):
    temp = []
    for y in range(1, inp+1):
        temp.append(y*i)
        ends = " "
    table.append(temp)
print(table)
print("hello word")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
death19 = pd.read_csv('time_series_covid19_deaths_global.csv')
death19

#membuat data asia tenggara
asia =  ['Kamboja','Laos','Myanmar','Thailand','Vietnam','Brunei','Filipina','Indonesia','Malaysia','Singapura','Timor-Leste']
df_asia = death19.loc[
                death19['Country/Region'].isin(asia)
                & death19['Province/State'].isna()]
#menghapus kolom Province/State, Lat,	Long
df_asia = df_asia.drop(['Province/State','Lat','Long'],axis = 1) 
df_asia

#mengubah index kolom menjadi negara
df_asia.rename(index=lambda x: df_asia.at[x,'Country/Region'], inplace=True) #mengubah index menjadi negara
#menghapus kolom Country/Region
df_asia = df_asia.drop(['Country/Region'],axis = 1)
df_asia

#tranpose dataframe
trans_df_asia = df_asia.transpose()
trans_df_asia
##mengubah index kolom menjadi format tanggal
trans_df_asia.index = pd.to_datetime(trans_df_asia.index)
trans_df_asia

#membuat line plot data meninggal kasus coivd di asia tenggara
trans_df_asia.plot(figsize=(19,10))
plt.title('perkembangan kematian covid-19 di Asia Tenggara')
plt.xlabel('Dates')
plt.ylabel('death covid')
plt.show()
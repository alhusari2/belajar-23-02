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

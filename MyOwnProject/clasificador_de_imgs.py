import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
def get_data(ruta):
	data_x=pd.read_csv(ruta+"X.csv",header=None)
	data_y=pd.read_csv(ruta+"Y.csv",header=None)
	data_p=pd.read_csv(ruta+"P.csv",header=None)
	X=data_x.iloc[:,:].values
	Y=data_y.iloc[:,:].values
	P=data_p.iloc[:,:].values
	return X,Y,P

def normalizar(X):
	mu=np.mean(X,axis=0)
	sigma=np.std(X,axis=0)
	Xnorm=(X-mu)/sigma
	return Xnorm,mu,sigma 

def convertir_vetor_matriz(vetor,tam_matriz):
	matriz=vetor.reshape(tam_matriz,tam_matriz)
	return matriz

def desenhar_grupo_imagens(imgs):
	num_imgs=imgs.shape[0]
	tam_img=int(np.sqrt(imgs.shape[1]))
	if(num_imgs>10):
		tam_grid_fil=int(np.floor(np.sqrt(num_imgs)))
		tam_grid_col=tam_grid_fil
	else:
		tam_grid_fil=1
		tam_grid_col=num_imgs

	fig,axis=plt.subplots(tam_grid_fil,tam_grid_col,figsize=(20,20),sharex=True, sharey=True)
	#fig.tight_layout()
	fig.tight_layout(pad=3.0)
	if(tam_grid_fil==1):
		for i in range(num_imgs):
			vetor_img=imgs[i,:]
			img=convertir_vetor_matriz(vetor_img,tam_img)
			axis[i].imshow(img,cmap="gray")
	else:
		k=0
		for i in range(tam_grid_fil):
			for j in range(tam_grid_col):
				if(k>num_imgs-1):
					break
				vetor_img=imgs[k,:]
				img=convertir_vetor_matriz(vetor_img,tam_img)
				axis[i,j].imshow(img,cmap="gray")
				axis[i,j].set_title("Img: "+str(k+1),size=5)
				#axis[i,j].axis("off")
				k=k+1
			if(k>num_imgs-1):
				break

ruta="datos/"
n=36
X,Y,P=get_data(ruta)
Xnorm,mu,sigma=normalizar(X)
U,D,VT=svd(Xnorm)
V=VT.T
desenhar_grupo_imagens(V[:,0:n].T)
plt.show()